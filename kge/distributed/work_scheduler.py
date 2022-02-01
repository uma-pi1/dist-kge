import math
import time
import numpy as np
import numba
import torch
from collections import deque, defaultdict
from copy import deepcopy
from kge.misc import set_seeds
from kge.distributed.stratification_schedule_creator import StratificationScheduleCreator
from torch import multiprocessing as mp
from torch import distributed as dist
from enum import IntEnum
from typing import Optional, Dict, Tuple
from .misc import get_min_rank, initialize_worker_groups, set_master_environment
from dataclasses import dataclass


TORCH_TO_NP_DTYPE = {
    torch.long: np.long,
    torch.int64: np.long,
    torch.int32: np.int32,
    torch.int: np.int32,
}


class SCHEDULER_CMDS(IntEnum):
    GET_WORK = 0
    WORK_DONE = 1
    WORK = 2
    NO_WORK = 3
    WAIT = 4
    BARRIER = 5
    SHUTDOWN = 6
    INIT_INFO = 7
    GET_INIT_WORK = 8
    GET_LOCAL_ENT = 9
    PRE_LOCALIZE_WORK = 10
    REGISTER_EVAL_RESULT = 11
    GET_EVAL_RESULT = 12

@dataclass
class WorkPackage:

    partition_id = None
    partition_data = None
    entities_in_partition = None
    relations_in_partition = None
    wait = False


class WorkScheduler(mp.get_context("fork").Process):
    def __init__(
        self,
        config,
        dataset,
    ):
        self._config_check(config)
        super(WorkScheduler, self).__init__(daemon=False, name="work-scheduler")
        self.config = config
        self.dataset = dataset
        self.min_rank = get_min_rank(config)
        self.rank = self.min_rank - 1
        self.num_clients = config.get("job.distributed.num_workers")
        self.world_size = self.num_clients + self.min_rank
        self.num_partitions = config.get("job.distributed.num_partitions")
        self.done_workers = []
        self.asking_workers = []
        self.work_to_do = deque(list(range(self.num_partitions)))
        self.wait_time = 0.4
        self.repartition_epoch = config.get("job.distributed.repartition_epoch")
        self.init_up_to_entity = -1
        self.num_processed_partitions = 0
        self.eval_hists = []
        if config.get("job.distributed.scheduler_data_type") not in ["int", "int32", "int64", "long"]:
            raise ValueError("Only long and int is supported as dtype for the scheduler communication")
        self.data_type = getattr(torch, config.get("job.distributed.scheduler_data_type"))
        if self.repartition_epoch:
            self.repartition_future = None
            self.repartition_worker_pool = None

    def _init_in_started_process(self):
        self.partitions = self._load_partitions(self.num_partitions)
        self._define_local_entities()

    def _define_local_entities(self):
        entity_keys = torch.arange(self.dataset.num_entities(), dtype=self.data_type)
        local_entities = entity_keys[torch.randperm(len(entity_keys))].chunk(self.num_clients)
        self.local_entities = dict(zip(range(self.min_rank, self.min_rank + self.num_clients), local_entities))

    def _config_check(self, config):
        if (
            config.get("job.distributed.entity_sync_level") == "partition"
            and not config.get("negative_sampling.sampling_type") == "pooled"
        ):
            raise ValueError(
                "entity sync level 'partition' only supported with 'pooled' sampling."
            )

    @staticmethod
    def create(
        config,
        dataset,
    ):
        if config.get("job.type") != "train":
            partition_type = "random"
        else:
            partition_type = config.get("job.distributed.partition_type")

        if partition_type == "random":
            return RandomWorkScheduler(config=config, dataset=dataset)
        elif partition_type == "relation":
            return RelationWorkScheduler(config=config, dataset=dataset)
        elif partition_type == "graph-cut":
            return GraphCutWorkScheduler(config=config, dataset=dataset)
        elif partition_type == "stratification":
            return StratificationWorkScheduler(config=config, dataset=dataset)
        elif partition_type == "super-stratification":
            return SuperStratificationWorkScheduler(config=config, dataset=dataset)
        elif partition_type == "random-stratification":
            return RandomStratificationWorkScheduler(config=config, dataset=dataset)
        else:
            raise NotImplementedError()

    def run(self):
        self._init_in_started_process()
        set_seeds(config=self.config)
        set_master_environment(self.config)
        # we have to have a huge timeout here, since it is only called after a complete
        #  epoch on a partition
        print("start scheduler with rank", self.rank, "world_size", self.world_size)
        # we need to create the worker group here as well it need to be defined in
        #  all processes
        initialize_worker_groups(self.config, self.rank)
        barrier_count = 0
        shutdown_count = 0
        epoch_time = None
        if self.repartition_epoch:
            if self.repartition_worker_pool is None:
                mp.Pool()
                self.repartition_worker_pool = mp.Pool(processes=1)
            self._repartition_in_background()

        while True:
            # cmd_buffer consists of cmd_number, key_len
            cmd_buffer = torch.full((2,), -1, dtype=self.data_type)

            # refill work and distribute to all asking workers
            if len(self.done_workers) == self.num_clients:
                epoch_time += time.time()
                self.config.log(f"complete_epoch_time: {epoch_time}")
                epoch_time = None
                self.num_processed_partitions = 0
                self._refill_work()
                for worker in self.asking_workers:
                    self._send_work(worker, cmd_buffer)
                self.done_workers = []
                self.asking_workers = []
                continue

            rank = dist.recv(cmd_buffer)
            cmd = cmd_buffer[0].item()
            if cmd == SCHEDULER_CMDS.GET_WORK:
                if epoch_time is None:
                    epoch_time = -time.time()
                if rank in self.done_workers:
                    self.asking_workers.append(rank)
                    continue
                machine_id = cmd_buffer[1].item()
                work_package = self._next_work(rank, machine_id)
                self._send_work(rank, cmd_buffer, work_package)
            elif cmd == SCHEDULER_CMDS.WORK_DONE:
                self._handle_work_done(rank)
            elif cmd == SCHEDULER_CMDS.BARRIER:
                barrier_count += 1
                if barrier_count == self.num_clients:
                    barrier_count = 0
                    dist.barrier()
            elif cmd == SCHEDULER_CMDS.SHUTDOWN:
                shutdown_count += 1
                if shutdown_count == self.num_clients:
                    print("shutting down work scheduler")
                    if self.repartition_epoch:
                        if self.repartition_worker_pool is not None:
                            self.repartition_worker_pool.close()
                            self.repartition_worker_pool.terminate()
                    break
            elif cmd == SCHEDULER_CMDS.INIT_INFO:
                self._handle_init_info(rank)
            elif cmd == SCHEDULER_CMDS.GET_INIT_WORK:
                self._handle_get_init_work(
                    rank=rank, embedding_layer_size=cmd_buffer[1].item()
                )
            elif cmd == SCHEDULER_CMDS.GET_LOCAL_ENT:
                self._handle_get_local_entities(rank=rank)
            elif cmd == SCHEDULER_CMDS.PRE_LOCALIZE_WORK:
                machine_id = cmd_buffer[1].item()
                work_package = self._handle_pre_localize_work(
                    rank=rank, machine_id=machine_id
                )
                self._send_work(
                    rank, cmd_buffer, work_package, pre_localize=True
                )
            elif cmd == SCHEDULER_CMDS.REGISTER_EVAL_RESULT:
                self._handle_register_eval_result(rank, cmd_buffer)
            elif cmd == SCHEDULER_CMDS.GET_EVAL_RESULT:
                self._handle_get_eval_result(rank)
            else:
                raise ValueError(
                    f"The work scheduler received an unknown command: {cmd}"
                )

    def _next_work(
        self, rank, machine_id
    ) -> WorkPackage:
        raise NotImplementedError()

    def _refill_work(self):
        self.work_to_do = deque(list(range(self.num_partitions)))

    def _repartition_in_background(self):
        pass

    def _send_work(
        self, rank, cmd_buffer, work_package, pre_localize=False
    ):
        # work, entities, relations, wait = self._next_work(rank)
        if work_package.partition_data is not None:
            cmd_buffer[0] = SCHEDULER_CMDS.WORK
            cmd_buffer[1] = len(work_package.partition_data)
            dist.send(cmd_buffer, dst=rank)
            dist.send(work_package.partition_data, dst=rank)
            if work_package.entities_in_partition is None:
                cmd_buffer[1] = 0
                dist.send(cmd_buffer, dst=rank)
            else:
                cmd_buffer[1] = len(work_package.entities_in_partition)
                dist.send(cmd_buffer, dst=rank)
                dist.send(work_package.entities_in_partition, dst=rank)
            if work_package.relations_in_partition is None:
                cmd_buffer[1] = 0
                dist.send(cmd_buffer, dst=rank)
            else:
                cmd_buffer[1] = len(work_package.relations_in_partition)
                dist.send(cmd_buffer, dst=rank)
                dist.send(work_package.relations_in_partition, dst=rank)
        elif work_package.wait:
            cmd_buffer[0] = SCHEDULER_CMDS.WAIT
            cmd_buffer[1] = self.wait_time
            dist.send(cmd_buffer, dst=rank)
        else:
            if not pre_localize:
                self.done_workers.append(rank)
            cmd_buffer[0] = SCHEDULER_CMDS.NO_WORK
            cmd_buffer[1] = 0
            dist.send(cmd_buffer, dst=rank)

    def _handle_work_done(self, rank):
        self.num_processed_partitions += 1
        print(f"trainer {rank} done with partition {self.num_processed_partitions}")

    def _handle_init_info(self, rank):
        max_entities = self._get_max_entities()
        max_relations = self._get_max_relations()
        init_data = torch.tensor([max_entities, max_relations], dtype=self.data_type)
        dist.send(init_data, dst=rank)

    def _handle_get_init_work(self, rank, embedding_layer_size):
        if self.init_up_to_entity == -1:
            print("initialize parameter server")
        self.init_up_to_entity += 1
        if self.init_up_to_entity >= self.dataset.num_entities():
            return_buffer = torch.tensor([-1, -1], dtype=self.data_type)
        else:
            entity_range_end = min(
                self.dataset.num_entities(),
                self.init_up_to_entity + embedding_layer_size,
            )
            if entity_range_end == self.dataset.num_entities():
                print("parameter server initialized")
            return_buffer = torch.tensor([self.init_up_to_entity, entity_range_end], dtype=self.data_type)
        self.init_up_to_entity += embedding_layer_size
        dist.send(return_buffer, dst=rank)

    def _handle_get_local_entities(self, rank):
        size_information = torch.tensor([len(self.local_entities[rank]), -1], dtype=self.data_type)
        dist.send(size_information, dst=rank)
        dist.send(self.local_entities[rank], dst=rank)

    def _handle_pre_localize_work(self, rank, machine_id):
        raise ValueError("The current partition scheme does not support pre-localizing")

    def _handle_register_eval_result(self, rank, cmd_buffer):
        num_sub_hists = cmd_buffer[1]
        first_eval = False
        if len(self.eval_hists) == 0:
            first_eval = True
        for j in range(num_sub_hists):
            ranks = torch.empty(self.dataset.num_entities())
            dist.recv(ranks, src=rank)
            if first_eval:
                self.eval_hists.append(ranks)
            else:
                self.eval_hists[j] += ranks

    def _handle_get_eval_result(self, rank):
        for i, h in enumerate(self.eval_hists):
            dist.send(h, dst=rank)
        self.eval_hists = []

    def _get_max_entities(self):
        return 0

    def _get_max_relations(self):
        return 0

    def _load_partitions(self, num_partitions):
        raise NotImplementedError()


class RandomWorkScheduler(WorkScheduler):
    def __init__(
        self,
        config,
        dataset,
    ):
        dataset._partition_type = "random"
        super(RandomWorkScheduler, self).__init__(
            config=config,
            dataset=dataset,
        )

    def _next_work(
        self, rank, machine_id
    ) -> WorkPackage:
        """add work/partitions to the list of work to do"""
        try:
            work_package = WorkPackage()
            work_package.partition_id = self.work_to_do.pop()
            work_package.partition_data = self.partitions[work_package.partition_id]
            # those are not entities in the partition but "local" entities for the
            #  worker to allow local sampling
            work_package.entities_in_partition = self.local_entities[rank]
            return work_package
        except IndexError:
            return WorkPackage()

    def _load_partitions(self, num_partitions):
        num_triples = len(self.dataset.split("train"))
        permuted_triple_index = torch.randperm(num_triples, dtype=self.data_type)
        partitions = list(torch.chunk(permuted_triple_index, num_partitions))
        partitions = [p.clone() for p in partitions]
        return partitions

    def _refill_work(self):
        if self.repartition_epoch:
            self.partitions = self._load_partitions(self.num_partitions)
            self._define_local_entities()
        super(RandomWorkScheduler, self)._refill_work()


class RelationWorkScheduler(WorkScheduler):
    def __init__(
        self,
        config,
        dataset,
    ):
        dataset._partition_type = "relation"
        super(RelationWorkScheduler, self).__init__(
            config=config,
            dataset=dataset,
        )

    def _init_in_started_process(self):
        super(RelationWorkScheduler, self)._init_in_started_process()
        self.relations_to_partition = self.dataset.load_relations_to_partitions(self.num_partitions)
        self.relations_to_partition = self._get_relations_in_partition()

    def _next_work(
        self, rank, machine_id
    ) -> WorkPackage:
        """add work/partitions to the list of work to do"""
        try:
            work_package = WorkPackage()
            work_package.partition_id = self.work_to_do.pop()
            work_package.partition_data = self.partitions[work_package.partition_id]
            work_package.relations_in_partition = self.relations_to_partition[work_package.partition_id]
            # those are not entities in the partition but "local" entities for the
            #  worker to allow local sampling
            work_package.entities_in_partition = self.local_entities[rank]
            return work_package
        except IndexError:
            return WorkPackage()

    def _load_partitions(self, num_partitions):
        np_type = TORCH_TO_NP_DTYPE[self.data_type]
        partition_assignment = self.dataset.load_train_partitions(num_partitions)
        # todo: let the partitions start at zero, then we do not need this unique
        partition_indexes = np.unique(partition_assignment)
        partitions = [
            torch.from_numpy(np.where(partition_assignment == i)[0].astype(np_type)).contiguous()
            for i in partition_indexes
        ]
        return partitions

    def _get_relations_in_partition(self):
        np_type = TORCH_TO_NP_DTYPE[self.data_type]
        relations_in_partition = dict()
        for partition in range(self.num_partitions):
            relations_in_partition[partition] = torch.from_numpy(
                np.where((self.relations_to_partition == partition),)[0].astype(np_type)
            ).contiguous()
        return relations_in_partition

    def _refill_work(self):
        if self.repartition_epoch:
            # self.partitions = self._load_partitions(self.num_partitions)
            self._define_local_entities()
        super(RelationWorkScheduler, self)._refill_work()


class GraphCutWorkScheduler(WorkScheduler):
    def __init__(
        self,
        config,
        dataset,
    ):
        dataset._partition_type = "graph-cut"
        super(GraphCutWorkScheduler, self).__init__(
            config=config,
            dataset=dataset,
        )

    def _init_in_started_process(self):
        super(GraphCutWorkScheduler, self)._init_in_started_process()
        self.entities_to_partition = self.dataset.load_entities_to_partitions(self.num_partitions)
        self.entities_to_partition = self._get_entities_in_partition()
        self.previous_partition_per_worker = defaultdict(lambda: None)

    def _config_check(self, config):
        super(GraphCutWorkScheduler, self)._config_check(config)
        if config.get("job.distributed.entity_sync_level") == "partition":
            raise ValueError(
                "Metis partitioning does not support entity sync level 'parititon'. "
                "Triples still have outside partition accesses."
            )

    def _next_work(
        self, rank, machine_id
    ) -> WorkPackage:
        """add work/partitions to the list of work to do"""
        try:
            work_package = WorkPackage()
            prev_work_id = self.previous_partition_per_worker[rank]
            if prev_work_id is not None and prev_work_id in self.work_to_do:
                work_package.partition_id = prev_work_id
                del self.work_to_do[self.work_to_do.index(prev_work_id)]
            else:
                work_package.partition_id = self.work_to_do.pop()
            work_package.partition_data = self.partitions[work_package.partition_id]
            work_package.entities_in_partition = self.entities_to_partition[work_package.partition_id]
            return work_package
        except IndexError:
            return WorkPackage()

    def _load_partitions(self, num_partitions):
        np_type = TORCH_TO_NP_DTYPE[self.data_type]
        partition_assignment = self.dataset.load_train_partitions(num_partitions)
        # todo: let the partitions start at zero, then we do not need this unique
        partition_indexes = np.unique(partition_assignment)
        partitions = [
            torch.from_numpy(np.where(partition_assignment == i)[0].astype(np_type)).contiguous()
            for i in partition_indexes
        ]
        return partitions

    def _get_entities_in_partition(self):
        np_type = TORCH_TO_NP_DTYPE[self.data_type]
        entities_in_partition = dict()
        for partition in range(self.num_partitions):
            entities_in_partition[partition] = torch.from_numpy(
                np.where((self.entities_to_partition == partition),)[0].astype(np_type)
            ).contiguous()
        return entities_in_partition

    def _get_max_entities(self):
        return max([len(i) for i in self.entities_to_partition.values()])


class StratificationWorkScheduler(WorkScheduler):
    """
    Lets look at the PBG scheduling here to make it correct
    """

    def __init__(
        self,
        config,
        dataset,
    ):
        dataset._partition_type = "stratification"
        self.combine_mirror_blocks = config.get("job.distributed.stratification.combine_mirror")
        super(StratificationWorkScheduler, self).__init__(
            config=config,
            dataset=dataset,
        )
        self.schedule_creator = StratificationScheduleCreator(
            num_partitions=self.num_partitions,
            num_workers=self.num_clients,
            randomize_iterations=True,
            combine_mirror_blocks=self.combine_mirror_blocks,
        )
        self.fixed_schedule = self.schedule_creator.create_schedule()
        self.current_iteration = set()
        self._pre_localized_strata: Dict[int, Tuple[int, int]] = {}
        # dictionary: key=worker_rank, value=block
        self.running_blocks: Dict[int, Tuple[int, int]] = {}
        self.entities_needed_only = self.config.get(
            "job.distributed.stratification.entities_needed_only"
        )
        self.num_max_entities = 0

    def _init_in_started_process(self):
        super(StratificationWorkScheduler, self)._init_in_started_process()
        # self.work_to_do = deepcopy(self.partitions)
        self._initialized_entity_blocks = set()
        entities_to_partition = self.dataset.load_entities_to_partitions(self.num_partitions)
        self._entities_in_bucket = self._get_entities_in_strata(
            entities_to_partition,
            self.partitions,
            self.dataset.split("train"),
            self.entities_needed_only,
            self.combine_mirror_blocks,
            TORCH_TO_NP_DTYPE[self.data_type]
        )
        if not self.fixed_schedule:
            self.work_to_do: Dict[Tuple[int, int], torch.Tensor] = self._order_by_schedule(
                deepcopy(self.partitions)
            )

    @staticmethod
    @numba.guvectorize(
        [(numba.int64[:], numba.int64, numba.int64, numba.int64[:])], "(n),(),()->(n)"
    )
    def _get_partition(entity_ids, num_entities, num_partitions, res):
        """
        This method maps a (already mapped) entity id to it's entity_partition.
        NOTE: you can not provide named parameters (kwargs) to this function
        Args:
            entity_ids: (mapped) entity ids np.array()
            num_entities: dataset.num_entities()
            num_partitions: int
            res: DON'T PROVIDE THIS. This is the resulting np.array of this vectorized
                function.

        Returns: np.array of entity ids mapped to partition

        """
        for i in range(len(entity_ids)):
            res[i] = math.floor(
                entity_ids[i] * 1.0 / num_entities * 1.0 * num_partitions
            )

    @staticmethod
    def _repartition(
        data,
        num_entities,
        num_partitions,
        entities_needed_only=True,
        combine_mirror_blocks=True,
        np_type=np.long,
    ):
        """
        This needs to be a static method so that we can pickle and run in background
        Args:
            data: data to repartition (train-set)
            num_entities: dataset.num_entities()
            num_partitions: self.num_partitions

        Returns:
            partitions: dict of structure {(block_id 1, block_id 2): [triple ids]}
            entities_in_bucket:
                dict of structure {(block_id 1, block_id 2): list of entity ids}
        """
        print("repartitioning data")
        start = -time.time()

        def random_map_entities():
            mapper = np.random.permutation(num_entities)
            mapped_data = deepcopy(data)  # drop reference to dataset
            mapped_data = mapped_data.numpy()
            mapped_data[:, 0] = mapper[mapped_data[:, 0]]
            mapped_data[:, 2] = mapper[mapped_data[:, 2]]
            return mapped_data, mapper

        mapped_data, mapped_entities = random_map_entities()
        print("repartition s")
        s_block = StratificationWorkScheduler._get_partition(
            mapped_data[:, 0], num_entities, num_partitions,
        )
        print("repartition o")
        o_block = StratificationWorkScheduler._get_partition(
            mapped_data[:, 2], num_entities, num_partitions,
        )
        print("map entity ids to partition")
        entity_to_partition = StratificationWorkScheduler._get_partition(
            mapped_entities, num_entities, num_partitions,
        )
        triple_partition_assignment = np.stack([s_block, o_block], axis=1)
        partitions = StratificationWorkScheduler._construct_partitions(
            triple_partition_assignment, num_partitions
        )
        entities_in_bucket = StratificationWorkScheduler._get_entities_in_strata(
            entity_to_partition,
            partitions,
            data.numpy(),
            entities_needed_only,
            combine_mirror_blocks,
            np_type
        )
        print("repartitioning done")
        print("repartition_time", start + time.time())
        return partitions, entities_in_bucket

    @staticmethod
    def _get_entities_in_strata(
        entities_to_partition,
        partitions,
        data,
        entities_needed_only,
        combine_mirror_blocks,
        np_type,
    ):
        entities_in_strata = dict()
        if entities_needed_only:
            for strata, strata_data in partitions.items():
                if combine_mirror_blocks:
                    if strata in entities_in_strata:
                        continue
                    if strata[0] == strata[1]:
                        if strata[0] % 2 == 0:
                            continue
                        mirror_strata = (strata[0] - 1, strata[1] - 1)
                    else:
                        mirror_strata = (strata[1], strata[0])
                    mirror_data = partitions[mirror_strata]
                    # for some reason torch.cat hangs on some machines on larger
                    # datasets when run in background, use numpy instead
                    combined_strata_data = np.concatenate((strata_data, mirror_data))
                    unique_entities = torch.from_numpy(
                        np.unique(data[combined_strata_data][:, [0, 2]]).astype(np_type)
                    ).contiguous()
                    entities_in_strata[strata] = unique_entities
                    entities_in_strata[mirror_strata] = unique_entities
                else:
                    # np.unique is slightly faster than torch.unique
                    entities_in_strata[strata] = torch.from_numpy(
                        np.unique(data[strata_data][:, [0, 2]]).astype(np_type)
                    ).contiguous()
        else:
            for strata in partitions.keys():
                if strata in entities_in_strata:
                    continue
                mirror_strata = (strata[1], strata[0])
                if combine_mirror_blocks:
                    if strata[0] == strata[1]:
                        if strata[0] % 2 == 0:
                            continue
                        mirror_strata = (strata[0] - 1, strata[1] - 1)
                entities = torch.from_numpy(
                    np.where(
                        np.ma.mask_or(
                            (entities_to_partition == strata[0]),
                            (entities_to_partition == mirror_strata[0]),
                        )
                    )[0].astype(np_type)
                ).contiguous()
                entities_in_strata[strata] = entities
                entities_in_strata[mirror_strata] = entities
        return entities_in_strata

    def _get_max_entities(self):
        if self.num_max_entities > 0:
            # store the result so that we don't have to recompute for every trainer
            return self.num_max_entities
        if self.entities_needed_only:
            num_entities_in_strata = [len(i) for i in self._entities_in_bucket.values()]
            len_std = np.std(num_entities_in_strata).item()
            if self.combine_mirror_blocks:
                max_num_entities, std_num_entities = self._get_mirrored_max_entities(
                    self.num_partitions,
                    list(self._entities_in_bucket.values()),
                    return_std=True,
                )
                self.num_max_entities = max_num_entities + 2 * (round(std_num_entities))
            else:
                self.num_max_entities = max(num_entities_in_strata) + 5 * round(len_std)
        else:
            self.num_max_entities = max(
                [len(i) for i in self._entities_in_bucket.values()]
            )
        return self.num_max_entities

    @staticmethod
    def _get_mirrored_max_entities(num_partitions, strata_entities, return_std=False):
        """
        Calculate how many entities occur at most if we combine mirrored blocks
        Combining blocks (0,1) and (1,0)
        For diagonals combine (0,0),(1,1), then (2,2),(3,3)...
        Count unique entities per combined block and return max
        Args:
            num_partitions: number of partitions
            strata_entities: list of unique entities occurring per strata
                assumes list is ordered

        Returns: max number of entities occurring in a combined mirror block

        """
        max_value = 0
        all_num_entities = []
        for i in range(num_partitions):
            for j in range(i, num_partitions):
                num_entities = 0
                # combine mirrored blocks
                if i % 2 == 0 and i == j:
                    # diagonal blocks: combine with following diagonal
                    concat_entities = np.concatenate(
                        (strata_entities[i], strata_entities[i + num_partitions])
                    )
                    num_entities = len(np.unique(concat_entities))
                elif i != j:
                    # combine (0,1) with (1,0) and so on
                    num_entities = len(
                        np.unique(
                            np.concatenate(
                                (
                                    strata_entities[i * num_partitions + j],
                                    strata_entities[j * num_partitions + i],
                                )
                            )
                        )
                    )
                all_num_entities.append(num_entities)
                if num_entities > max_value:
                    # this will lead to a race condition if we do this in parallel
                    max_value = num_entities
        all_num_entities = np.array(all_num_entities)
        max_value = all_num_entities.max()
        if return_std:
            std = all_num_entities.std()
            return max_value, std
        print("max entities", max_value)
        return max_value

    def _next_work(
        self, rank, machine_id
    ) -> WorkPackage:
        return self._acquire_strata(rank, machine_id)

    def _handle_pre_localize_work(self, rank, machine_id):
        return self._acquire_strata(rank, machine_id, pre_localize=True)

    def _acquire_strata(self, rank, machine_id, pre_localize=False):
        try:
            if len(self.current_iteration) == 0:
                self.current_iteration = set(self.fixed_schedule.pop())
        except IndexError:
            return WorkPackage()
        return self._acquire_strata_by_schedule(
            rank, current_iteration=self.current_iteration, pre_localize=pre_localize
        )

    def _acquire_strata_by_schedule(self, rank, current_iteration, pre_localize=False):
        work_package = WorkPackage()
        try:
            locked_entity_strata = set()
            for locked_dict in [self.running_blocks, self._pre_localized_strata]:
                for running_rank, strata in locked_dict.items():
                    if rank == running_rank:
                        continue
                    locked_entity_strata.add(strata[0])
                    locked_entity_strata.add(strata[1])

            def _strata_locked(strata):
                return (
                    strata[0] in locked_entity_strata
                    or strata[1] in locked_entity_strata
                )

            def _acquire(strata, acquire_pre_localized=False):
                if acquire_pre_localized:
                    del self._pre_localized_strata[rank]
                else:
                    current_iteration.remove(strata)
                strata_data = self.partitions[strata]
                entities_in_strata = self._entities_in_bucket.get(strata)
                if self.combine_mirror_blocks and strata_data is not None:
                    if strata[0] == strata[1]:
                        mirror_strata = (strata[0] - 1, strata[1] - 1)
                    else:
                        mirror_strata = (strata[1], strata[0])
                    strata_data = torch.cat(
                        (strata_data, self.partitions[mirror_strata])
                    )
                if not pre_localize:
                    self.running_blocks[rank] = strata
                else:
                    self._pre_localized_strata[rank] = strata
                work_package.partition_id = strata
                work_package.partition_data = strata_data
                work_package.entities_in_partition = entities_in_strata
                return work_package

            # only use pre localized strata, if we are not about to pre-localize a new
            # one --> not pre_localize
            if (
                not pre_localize
                and self._pre_localized_strata.get(rank, None) is not None
            ):
                strata = self._pre_localized_strata[rank]
                if _strata_locked(strata):
                    # we are waiting until the localized strata is free
                    work_package.wait = True
                    return work_package
                return _acquire(strata, acquire_pre_localized=True)

            for strata in current_iteration:
                if _strata_locked(strata):
                    continue
                return _acquire(strata)

            # return wait here
            work_package.wait = True
            return work_package
        except IndexError:
            return work_package

    def _handle_work_done(self, rank):
        super(StratificationWorkScheduler, self)._handle_work_done(rank)
        del self.running_blocks[rank]

    def _repartition_in_background(self):
        self.repartition_future = self.repartition_worker_pool.apply_async(
            self._repartition,
            (
                self.dataset.split("train"),
                self.dataset.num_entities(),
                self.num_partitions,
                self.entities_needed_only,
                self.combine_mirror_blocks,
                TORCH_TO_NP_DTYPE[self.data_type],
            ),
        )

    def _refill_work(self):
        if self.repartition_epoch:
            self.partitions, self._entities_in_bucket = self.repartition_future.get()
            self._repartition_in_background()
        self.fixed_schedule = self.schedule_creator.create_schedule()
        if self.fixed_schedule is None:
            self.work_to_do = self._order_by_schedule(deepcopy(self.partitions))

    def _load_partitions(self, num_partitions):
        start = time.time()
        partition_assignment = self.dataset.load_train_partitions(num_partitions)
        partitions = self._construct_partitions(partition_assignment, num_partitions)
        print("partition load time", time.time() - start)
        return partitions

    def _construct_partitions(self, partition_assignment, num_partitions):
        (
            partition_indexes,
            partition_data,
        ) = StratificationWorkScheduler._numba_construct_partitions(
            np.ascontiguousarray(partition_assignment), num_partitions
        )
        partition_indexes = [
            (i, j) for i in range(num_partitions) for j in range(num_partitions)
        ]
        partition_data = [
            torch.from_numpy(data).to(self.data_type).contiguous() for data in partition_data
        ]
        partitions = dict(zip(partition_indexes, partition_data))
        return partitions

    @staticmethod
    @numba.njit
    def _numba_construct_partitions(partition_assignment, num_partitions):
        partition_indexes = [
            (i, j) for i in range(num_partitions) for j in range(num_partitions)
        ]
        partition_id_lookup: Dict[Tuple[int, int], int] = dict()
        partition_lengths: Dict[int, int] = dict()
        partition_data = []
        for i in range(len(partition_indexes)):
            partition = partition_indexes[i]
            partition_id_lookup[partition] = i
            partition_lengths[i] = 0
            partition_data.append(
                np.empty(
                    int(len(partition_assignment) / num_partitions), dtype=np.int64
                )
            )

        # iterate over the partition assignments and assign each triple-id to its
        #  corresponding partition
        for i in range(len(partition_assignment)):
            pa = partition_assignment[i]
            pa_tuple = (pa[0], pa[1])
            partition_id = partition_id_lookup[pa_tuple]
            current_partition_size = partition_lengths[partition_id]
            partition_data[partition_id][current_partition_size] = i
            partition_lengths[partition_id] += 1

        # now get correct sizes of partitions
        for i in range(len(partition_data)):
            partition_data[i] = partition_data[i][: partition_lengths[i]]
        return partition_indexes, partition_data


class SuperStratificationWorkScheduler(StratificationWorkScheduler):
    """
    This changes the schedule of the StratificationWorkScheduler
    Build a super schedule per machine (num_machines*2xnum_machines*2)
    Build a sub-schedule for each super schedule
    """
    def __init__(
            self,
            config,
            dataset,
    ):
        # create a super scheduler with num_machines*4 partitions
        self.num_machines = config.get("job.distributed.num_machines")
        self.num_super_partitions = self.num_machines*2
        super(SuperStratificationWorkScheduler, self).__init__(config=config, dataset=dataset)

    def _init_in_started_process(self):
        super(SuperStratificationWorkScheduler, self)._init_in_started_process()
        self.super_stratification_scheduler = StratificationWorkScheduler(config=self.config, dataset=self.dataset)
        self.super_stratification_scheduler.partitions = defaultdict(lambda: None)
        self.super_stratification_scheduler._entities_in_bucket = defaultdict(lambda: None)
        self.current_machine_strata: Dict[Optional[Tuple[int, int]]] = defaultdict(lambda: None)
        self.current_machine_iteration = defaultdict(set)
        # fixme: here we assume each machine has the same amount of workers
        #  this is not necessarily true
        num_workers_per_machine = int(self.config.get("job.distributed.num_workers")/self.config.get("job.distributed.num_machines"))
        self.schedule_creator = StratificationScheduleCreator(
            num_partitions=int(self.num_partitions/self.num_super_partitions),
            num_workers=num_workers_per_machine,
            randomize_iterations=True,
            # combine mirror is False, since default scheduler already combines
            combine_mirror_blocks=False,
        )
        self._create_schedule_per_super_strata()

    def _create_schedule_per_super_strata(self):
        self.schedule_per_super_strata = dict()
        for i in range(self.num_super_partitions):
            self.schedule_creator.i_offset = i * self.num_super_partitions
            for j in range(self.num_super_partitions):
                self.schedule_creator.j_offset = j * self.num_super_partitions
                self.schedule_per_super_strata[(i, j)] = self.schedule_creator.create_schedule()

    def _acquire_strata(self, rank, machine_id, pre_localize=False):
        if self.current_machine_strata[machine_id] is None:
            self.current_machine_strata[machine_id] = self.super_stratification_scheduler._acquire_strata(rank=machine_id, machine_id=machine_id, pre_localize=False)
        if self.current_machine_strata[machine_id].partition_id is None:
            # we are done with all super strata
            self.current_machine_strata[machine_id] = None
            return WorkPackage()
        try:
            if len(self.current_machine_iteration[machine_id]) == 0:
                if self.current_machine_strata[machine_id].partition_id is None:
                    print("somethings off")
                self.current_machine_iteration[machine_id] = set(
                    self.schedule_per_super_strata[self.current_machine_strata[machine_id].partition_id].pop()
                )
        except IndexError:
            # this super strata is done, get a new super strata
            self.current_machine_strata[machine_id] = None
            self.super_stratification_scheduler._handle_work_done(rank=machine_id)
            return self._acquire_strata(rank, machine_id, pre_localize=pre_localize)
        # now get the actual sub-strata
        return self._acquire_strata_by_schedule(
            rank,
            current_iteration=self.current_machine_iteration[machine_id],
            pre_localize=pre_localize
        )

    def _refill_work(self):
        super(SuperStratificationWorkScheduler, self)._refill_work()
        self.super_stratification_scheduler.fixed_schedule = self.super_stratification_scheduler.schedule_creator.create_schedule()
        self._create_schedule_per_super_strata()


class RandomStratificationWorkScheduler(StratificationWorkScheduler):
    def __init__(
            self,
            config,
            dataset,
    ):
        # num_partitions = self.config.get("job.distributed.num_machines")*2
        # num_clients = self.config.get("job.distributed.num_machines")
        super(RandomStratificationWorkScheduler, self).__init__(config, dataset)
        self.num_machines = self.config.get("job.distributed.num_machines")
        self.num_partitions = self.num_machines * 4
        # override the schedule creator
        self.schedule_creator = StratificationScheduleCreator(
            num_partitions=self.num_partitions,
            num_workers=self.num_machines,
            randomize_iterations=True,
            combine_mirror_blocks=self.combine_mirror_blocks,
        )
        self.fixed_schedule = self.schedule_creator.create_schedule()
        self.work_to_do_per_machine = defaultdict(list)
        # todo: find out how many worker we have per machine
        #  for now assume the same amount of worker per machine
        num_workers_machine = self.config.get("job.distributed.num_workers_machine")
        if num_workers_machine < 1:
            num_workers_machine = self.config.get("job.distributed.num_workers")
        self.num_workers_per_machine = defaultdict(lambda: num_workers_machine)

    def _next_work(
        self, rank, machine_id
    ) -> WorkPackage:
        if len(self.work_to_do_per_machine[machine_id]) == 0:
            if machine_id in self.running_blocks:
                del self.running_blocks[machine_id]
            work_package = super(RandomStratificationWorkScheduler, self)._next_work(
                rank=machine_id, machine_id=machine_id
            )
            if work_package.partition_data is None:
                return work_package
            partition_data_chunks = torch.chunk(
                work_package.partition_data[
                    torch.randperm(len(work_package.partition_data))
                ],
                self.num_workers_per_machine[machine_id]
            )
            for partition_data in partition_data_chunks:
                wp = WorkPackage()
                wp.partition_data = partition_data
                wp.entities_in_partition = work_package.entities_in_partition
                self.work_to_do_per_machine[machine_id].append(wp)
        return self.work_to_do_per_machine[machine_id].pop()

    def _handle_work_done(self, rank):
        self.num_processed_partitions += 1
        print(f"trainer {rank} done with partition {self.num_processed_partitions}")
        # handled in next work
        return


class SchedulerClient:
    def __init__(self, config):
        self.scheduler_rank = get_min_rank(config) - 1
        self.machine_id = config.get("job.distributed.machine_id")
        if config.get("job.distributed.scheduler_data_type") not in ["int", "int32", "int64", "long"]:
            raise ValueError("Only long and int is supported as dtype for the scheduler communication")
        self.data_type = getattr(torch, config.get("job.distributed.scheduler_data_type"))

    def get_init_info(self):
        cmd = torch.tensor([SCHEDULER_CMDS.INIT_INFO, 0], dtype=self.data_type)
        dist.send(cmd, dst=self.scheduler_rank)
        info_buffer = torch.zeros((2,), dtype=self.data_type)
        dist.recv(info_buffer, src=self.scheduler_rank)
        max_entities = info_buffer[0]
        max_relations = info_buffer[1]
        return max_entities, max_relations

    def _receive_work(self, cmd):
        work_buffer = torch.empty((cmd[1].item(),), dtype=self.data_type)
        dist.recv(work_buffer, src=self.scheduler_rank)
        # get partition entities
        dist.recv(cmd, src=self.scheduler_rank)
        num_entities = cmd[1].item()
        entity_buffer = None
        if num_entities != 0:
            entity_buffer = torch.empty((num_entities,), dtype=self.data_type)
            dist.recv(entity_buffer, src=self.scheduler_rank)
        # get partition relations
        dist.recv(cmd, src=self.scheduler_rank)
        num_relations = cmd[1].item()
        relation_buffer = None
        if num_relations != 0:
            relation_buffer = torch.empty((num_relations,), dtype=self.data_type)
            dist.recv(relation_buffer, src=self.scheduler_rank)
        return work_buffer, entity_buffer, relation_buffer

    def get_work(
        self,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        while True:
            cmd = torch.tensor([SCHEDULER_CMDS.GET_WORK, self.machine_id], dtype=self.data_type)
            dist.send(cmd, dst=self.scheduler_rank)
            dist.recv(cmd, src=self.scheduler_rank)
            if cmd[0] == SCHEDULER_CMDS.WORK:
                return self._receive_work(cmd)
            elif cmd[0] == SCHEDULER_CMDS.WAIT:
                # print("waiting for a block")
                time.sleep(cmd[1].item())
            else:
                return None, None, None

    def get_pre_localize_work(self):
        cmd = torch.tensor([SCHEDULER_CMDS.PRE_LOCALIZE_WORK, self.machine_id], dtype=self.data_type)
        dist.send(cmd, dst=self.scheduler_rank)
        dist.recv(cmd, src=self.scheduler_rank)
        if cmd[0] == SCHEDULER_CMDS.WORK:
            work, entities, relations = self._receive_work(cmd)
            return work, entities, relations, False
        elif cmd[0] == SCHEDULER_CMDS.WAIT:
            return None, None, None, True
        else:
            return None, None, None, False

    def get_init_work(self, entity_embedder_size):
        """
        Get the entity ids that should be initialized by the worker.
        Receives start and end id from the scheduler
        Args:
            entity_embedder_size: size of the local entity embedding layer

        Returns:
            tensor containing range from start and end entity id

        """
        cmd = torch.tensor([SCHEDULER_CMDS.GET_INIT_WORK, entity_embedder_size], dtype=self.data_type)
        dist.send(cmd, dst=self.scheduler_rank)
        dist.recv(cmd, src=self.scheduler_rank)
        if cmd[0] > -1:
            return torch.arange(cmd[0], cmd[1], dtype=self.data_type)
        return None

    def register_eval_result(self, hist: dict, hist_filt: dict, hist_filt_test: dict):
        hists = [hist, hist_filt, hist_filt_test]
        cmd = torch.tensor([SCHEDULER_CMDS.REGISTER_EVAL_RESULT, sum(len(h) for h in hists)], dtype=self.data_type)
        dist.send(cmd, dst=self.scheduler_rank)
        for h in hists:
            for v in h.values():
                ranks = v.cpu()
                dist.send(ranks, dst=self.scheduler_rank)

    def get_eval_result(self, hist: dict, hist_filt: dict, hist_filt_test: dict):
        cmd = torch.tensor([SCHEDULER_CMDS.GET_EVAL_RESULT, -1], dtype=self.data_type)
        dist.send(cmd, dst=self.scheduler_rank)
        hists = [hist, hist_filt, hist_filt_test]
        for h in hists:
            for key, values in h.items():
                ranks = torch.empty(len(values))
                dist.recv(ranks, src=self.scheduler_rank)
                h[key] = ranks
        return hist, hist_filt, hist_filt_test

    def get_local_entities(self):
        cmd = torch.tensor([SCHEDULER_CMDS.GET_LOCAL_ENT, -1], dtype=self.data_type)
        dist.send(cmd, dst=self.scheduler_rank)
        dist.recv(cmd, src=self.scheduler_rank)
        if cmd[0] > 0:
            local_entities = torch.empty([cmd[0], ], dtype=self.data_type)
            dist.recv(local_entities, src=self.scheduler_rank)
            return local_entities.long()
        return None

    def work_done(self):
        cmd = torch.tensor([SCHEDULER_CMDS.WORK_DONE, self.machine_id], dtype=self.data_type)
        dist.send(cmd, dst=self.scheduler_rank)

    def shutdown(self):
        cmd = torch.tensor([SCHEDULER_CMDS.SHUTDOWN, self.machine_id], dtype=self.data_type)
        dist.send(cmd, dst=self.scheduler_rank)
