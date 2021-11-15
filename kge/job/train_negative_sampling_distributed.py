import time
import shutil
import torch
import torch.utils.data
import numpy as np
import math
import gc
import os
import itertools

from collections import defaultdict, deque
from typing import Dict, Any

from kge.job import Job
from kge.job.train import TrainingJob, _generate_worker_init_fn
from kge.job.train_negative_sampling import TrainingJobNegativeSampling
from kge.model import KgeModel
from kge.util import KgeOptimizer
from kge.job.trace import format_trace_entry
from kge.distributed.work_scheduler import SchedulerClient
from kge.distributed.misc import get_min_rank

SLOTS = [0, 1, 2]
S, P, O = SLOTS
SLOT_STR = ["s", "p", "o"]


class NumberDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, dataset):
        self.samples = list(range(num_samples))
        self.dataset = dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return idx
        # return self.dataset[self.samples[idx], :].long()

    def set_samples(self, samples):
        self.samples = samples


class InfiniteSequentialSampler(torch.utils.data.Sampler):
    def __init__(self, data_source):
        super(InfiniteSequentialSampler, self).__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        return itertools.count(start=0, step=1)

    def __len__(self):
        return len(self.data_source)


class BatchDataset(torch.utils.data.Dataset):
    def __init__(self, triples, batch_size, shuffle=True):
        self.triples = triples
        # work around for now to have a working shared tensor
        self.samples = torch.empty([len(triples), ], dtype=torch.int, requires_grad=False).share_memory_()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = torch.full([1, ], -1, dtype=torch.int, requires_grad=False).share_memory_()
        self.epoch = torch.full([1, ], -1, dtype=torch.int, requires_grad=False).share_memory_()
        self.partition_id = torch.full([1, ], -1, dtype=torch.int, requires_grad=False).share_memory_()

    def __len__(self):
        if self.num_samples.item() <= 0:
            return 0
        return math.ceil(self.num_samples.item() / self.batch_size)

    def get_real_len(self):
        return math.ceil(self.num_samples.item() / self.batch_size)

    def __getitem__(self, idx):
        """Gets a complete batch based on an idx"""
        # we are iterating with a infinite sampler. Get the actual batch index
        # with modulo
        actual_idx = idx % len(self)
        start = actual_idx * self.batch_size
        stop = min((actual_idx + 1) * (self.batch_size), self.num_samples.item())
        if start >= stop:
            print(idx, self.num_samples.item(), start, stop, len(self))
            return None
        return (self.samples[
            start: stop
        ].clone().long(), self.epoch.item(), self.partition_id.item())

    def set_samples(self, samples: torch.Tensor, epoch, partition_id):
        if self.shuffle:
            samples = samples.numpy()
            np.random.shuffle(samples)
            samples = torch.from_numpy(samples)
        self.samples[:len(samples)] = samples
        self.num_samples[0] = len(samples)
        self.epoch[0] = epoch
        self.partition_id[0] = partition_id


class TrainingJobNegativeSamplingDistributed(TrainingJobNegativeSampling):
    def __init__(
        self,
        config,
        dataset,
        parent_job=None,
        model=None,
        optimizer=None,
        forward_only=False,
        parameter_client=None,
        work_scheduler_client=None,
        init_for_load_only=False,
    ):
        self.parameter_client = parameter_client
        self.min_rank = get_min_rank(config)

        if work_scheduler_client is None:
            self.work_scheduler_client = SchedulerClient(config)
        else:
            self.work_scheduler_client = work_scheduler_client
        (
            max_partition_entities,
            max_partition_relations,
        ) = self.work_scheduler_client.get_init_info()
        if model is None:
            model: KgeModel = KgeModel.create(
                config,
                dataset,
                parameter_client=parameter_client,
                max_partition_entities=max_partition_entities,
            )
        model.get_s_embedder().to_device()
        model.get_p_embedder().to_device()
        lapse_indexes = [
            torch.arange(dataset.num_entities(), dtype=torch.int),
            torch.arange(dataset.num_relations(), dtype=torch.int)
            + dataset.num_entities(),
        ]
        if optimizer is None:
            optimizer = KgeOptimizer.create(
                config,
                model,
                parameter_client=parameter_client,
                lapse_indexes=lapse_indexes,
            )
        # barrier to wait for loading of pretrained embeddings
        self.parameter_client.barrier()
        super().__init__(
            config,
            dataset,
            parent_job,
            model=model,
            optimizer=optimizer,
            forward_only=forward_only,
            parameter_client=parameter_client,
            work_scheduler_client=work_scheduler_client,
        )
        self.type_str = "negative_sampling"
        self.entity_localize = self.config.get("job.distributed.entity_localize")
        self.relation_localize = self.config.get("job.distributed.relation_localize")
        self.entity_partition_localized = False
        self.relation_partition_localized = False
        self.local_entities = None
        self.entity_async_write_back = self.config.get(
            "job.distributed.entity_async_write_back"
        )
        self.relation_async_write_back = self.config.get(
            "job.distributed.relation_async_write_back"
        )
        self.entity_sync_level = self.config.get("job.distributed.entity_sync_level")
        self.relation_sync_level = self.config.get(
            "job.distributed.relation_sync_level"
        )
        self.entity_pre_pull = self.config.get("job.distributed.entity_pre_pull")
        self.relation_pre_pull = self.config.get("job.distributed.relation_pre_pull")
        self.pre_localize_batch = int(
            self.config.get("job.distributed.pre_localize_batch")
        )
        self.entity_mapper_tensors = deque()
        for i in range(self.config.get("train.num_workers") + 1):
            self.entity_mapper_tensors.append(
                torch.full((self.dataset.num_entities(),), -1, dtype=torch.long)
            )

        # also defines the local entities
        self._initialize_parameter_server(init_for_load_only=init_for_load_only)

        def stop_and_wait(job):
            job.parameter_client.stop()
            job.parameter_client.barrier()
        self.early_stop_hooks.append(stop_and_wait)

        def check_stopped(job):
            print("checking for", job.parameter_client.rank)
            job.parameter_client.barrier()
            return job.parameter_client.is_stopped()
        self.early_stop_conditions.append(check_stopped)
        self.work_pre_localized = False
        if self.config.get("job.distributed.pre_localize_partition"):
            self.pre_localized_entities = None
            self.pre_localized_relations = None
            self.pre_batch_hooks.append(self._pre_localize_work)

        if self.__class__ == TrainingJobNegativeSamplingDistributed:
            for f in Job.job_created_hooks:
                f(self)

    def _initialize_parameter_server(self, init_for_load_only=False):
        # initialize the parameter server
        #  each worker takes as many entities as it can fit, inits and pushes
        #  init work is distributed by the work scheduler
        if not init_for_load_only and not self.config.get(
            "lookup_embedder.pretrain.model_filename"
        ):
            # only the first worker initializes the relations
            if self.parameter_client.rank == self.min_rank:
                self.model.get_p_embedder().push_all()
            entity_embedding_layer_size = self.model.get_s_embedder()._embeddings.weight.data.shape[0]
            self.local_entities = self.work_scheduler_client.get_local_entities()
            self.parameter_client.localize(self.local_entities, asynchronous=True)
            init_work_packages = self.local_entities.split(entity_embedding_layer_size)
            for init_work_package in init_work_packages:
                self.model.get_s_embedder().initialize(
                    self.model.get_s_embedder()._embeddings.weight.data
                )
                self.model.get_s_embedder()._normalize_embeddings()
                self._push_init_to_parameter_server(init_work_package)
        self.parameter_client.barrier()

    def _push_init_to_parameter_server(self, entity_ids: torch.Tensor):
        push_tensor = torch.cat(
            (
                self.model.get_s_embedder()
                ._embeddings.weight.data[: len(entity_ids)]
                .cpu(),
                self.model.get_s_embedder().optimizer_values[: len(entity_ids)].cpu(),
            ),
            dim=1,
        )
        self.parameter_client.push(
            entity_ids + self.model.get_s_embedder().lapse_offset, push_tensor.cpu(),
        )

    @staticmethod
    def _pre_localize_work(job, batch_index):
        if batch_index % 100 != 0:
            return
        if not job.work_pre_localized:
            work, entities, relations, wait = job.work_scheduler_client.get_pre_localize_work()
            if wait:
                return
            if entities is not None:
                entities_ps_offset = job.model.get_s_embedder().lapse_offset
                job.pre_localized_entities = entities + entities_ps_offset
                job.parameter_client.localize(
                    job.pre_localized_entities, asynchronous=True
                )
            if relations is not None:
                relations_ps_offset = job.model.get_p_embedder().lapse_offset
                job.pre_localized_relations = relations + relations_ps_offset
                job.parameter_client.localize(
                    job.pre_localized_relations, asynchronous=True
                )
            job.work_pre_localized = True

    def _prepare(self):
        """Construct dataloader"""
        super()._prepare()

        self.num_examples = self.dataset.split(self.train_split).size(0)
        self.dataloader_dataset = BatchDataset(
            self.dataset.split(self.train_split),
            batch_size=self.batch_size,
            shuffle=True,
        )
        # initializing dataloader as soon as we got the triples from work scheduler
        self.loader = None
        if self.config.get("negative_sampling.sampling_type") == "pooled":
            if self.local_entities is None:
                self.local_entities = self.work_scheduler_client.get_local_entities()
                self.parameter_client.localize(self.local_entities)
            self._sampler.set_pool(self.local_entities, S)
            self._sampler.set_pool(self.local_entities, O)

    def _get_collate_fun(self):
        # create the collate function
        def collate(batch):
            """For a batch of size n, returns a tuple of:

            - triples (tensor of shape [n,3], ),
            - negative_samples (list of tensors of shape [n,num_samples]; 3 elements
              in order S,P,O)
            """

            if batch[0] is None:
                # this can happen due to keeping the dataloader alive
                return None
            triple_ids = batch[0][0]
            epoch = batch[0][1]
            local_partition_id = batch[0][2]
            triples = self.dataset.split(self.train_split)[triple_ids, :].long()

            negative_samples = list()
            for slot in [S, P, O]:
                negative_samples.append(self._sampler.sample(triples, slot))
            unique_time = -time.time()
            unique_entities = torch.unique(
                torch.cat(
                    (
                        triples[:, [S, O]].view(-1),
                        negative_samples[S].unique_samples(remove_dropped=False),
                        negative_samples[O].unique_samples(remove_dropped=False),
                    )
                )
            )
            unique_relations = torch.unique(
                torch.cat(
                    (
                        triples[:, [P]].view(-1),
                        negative_samples[P].unique_samples(remove_dropped=False),
                    )
                )
            )
            unique_time += time.time()

            return {
                "triples": triples,
                "negative_samples": negative_samples,
                "unique_entities": unique_entities,
                "unique_relations": unique_relations,
                "unique_time": unique_time,
                "epoch": epoch,
                "local_partition_id": local_partition_id,
            }

        return collate

    def _map_ids_to_local(self, batch):

        # map ids to local ids
        if self.entity_sync_level == "partition":
            entity_mapper = self.model.get_s_embedder().global_to_local_mapper
        else:
            # entity_mapper = torch.full((self.dataset.num_entities(),), -1, dtype=torch.long)
            entity_mapper = self.entity_mapper_tensors.popleft()
            entity_mapper[batch["unique_entities"]] = torch.arange(
                len(batch["unique_entities"]), dtype=torch.long
            )
        if self.relation_sync_level == "partition":
            relation_mapper = self.model.get_p_embedder().global_to_local_mapper
        else:
            relation_mapper = torch.full(
                (self.dataset.num_relations(),), -1, dtype=torch.long
            )
            relation_mapper[batch["unique_relations"]] = torch.arange(
                len(batch["unique_relations"]), dtype=torch.long
            )
        batch["triples"][:, S] = entity_mapper[batch["triples"][:, S]]
        batch["triples"][:, P] = relation_mapper[batch["triples"][:, P]]
        batch["triples"][:, O] = entity_mapper[batch["triples"][:, O]]
        batch["negative_samples"][S].map_samples(entity_mapper)
        batch["negative_samples"][P].map_samples(relation_mapper)
        batch["negative_samples"][O].map_samples(entity_mapper)

        # for debugging reset the entity mapper to -1
        # entity_mapper[:] = -1
        self.entity_mapper_tensors.append(entity_mapper)
        return batch

    def _prepare_batch_ahead(self, batches: deque):
        if self.entity_pre_pull > 1 or self.relation_pre_pull > 1:
            batches[0]["triples"] = batches[0]["triples"].to(self.device)
            for ns in batches[0]["negative_samples"]:
                ns.positive_triples = batches[0]["triples"]
            batches[0]["negative_samples"] = [
                ns.to(self.device) for ns in batches[0]["negative_samples"]
            ]
        if self.entity_sync_level == "batch" and self.entity_pre_pull > 0:
            self.model.get_s_embedder().pre_pull(batches[-1]["unique_entities"])
            self.model.get_s_embedder().pre_pulled_to_device()
        if self.relation_sync_level == "batch" and self.relation_pre_pull > 0:
            self.model.get_p_embedder().pre_pull(batches[-1]["unique_relations"])
            self.model.get_p_embedder().pre_pulled_to_device()
        if self.pre_localize_batch > 0:
            self.model.get_s_embedder().localize(
                batches[-1]["unique_entities"],
                asynchronous=True
            )

    def _prepare_batch(
        self, batch_index, batch, result: TrainingJob._ProcessBatchResult
    ):
        # move triples and negatives to GPU. With some implementaiton effort, this may
        # be avoided.
        result.prepare_time -= time.time()
        # result.cpu_gpu_time -= time.time()
        batch["triples"] = batch["triples"].to(self.device)
        for ns in batch["negative_samples"]:
            ns.positive_triples = batch["triples"]
        batch["negative_samples"] = [
            ns.to(self.device) for ns in batch["negative_samples"]
        ]
        # result.cpu_gpu_time += time.time()
        result.unique_time += batch["unique_time"]
        if self.entity_sync_level == "batch":
            unique_entities = batch["unique_entities"]

            result.ps_wait_time -= time.time()
            if not self.entity_async_write_back:
                for wait_value in self.optimizer.entity_async_wait_values:
                    self.parameter_client.wait(wait_value)
                self.optimizer.entity_async_wait_values.clear()
            result.ps_wait_time += time.time()
            if self.entity_localize and not self.entity_partition_localized:
                self.model.get_s_embedder().localize(
                    unique_entities, asynchronous=True
                )
            result.pull_and_map_time -= time.time()
            (
                entity_pull_time,
                cpu_gpu_time,
            ) = self.model.get_s_embedder()._pull_embeddings(unique_entities)
            result.pull_and_map_time += time.time()
            result.entity_pull_time += entity_pull_time
            result.cpu_gpu_time += cpu_gpu_time
        if self.relation_sync_level == "batch":
            unique_relations = batch["unique_relations"]
            result.ps_wait_time -= time.time()
            if not self.relation_async_write_back:
                for wait_value in self.optimizer.relation_async_wait_values:
                    self.parameter_client.wait(wait_value)
                self.optimizer.relation_async_wait_values.clear()
            result.ps_wait_time += time.time()
            if self.relation_localize and not self.relation_partition_localized:
                self.model.get_p_embedder().localize(
                    unique_relations, asynchronous=True
                )
            result.pull_and_map_time -= time.time()
            (
                relation_pull_time,
                cpu_gpu_time,
            ) = self.model.get_p_embedder()._pull_embeddings(unique_relations)
            result.pull_and_map_time += time.time()
            result.relation_pull_time += relation_pull_time
            result.cpu_gpu_time += cpu_gpu_time

        batch["labels"] = [None] * 3  # reuse label tensors b/w subbatches
        result.size = len(batch["triples"])
        result.prepare_time += time.time()

    def handle_validation(self, metric_name):
        # move all models to cpu and store as tmp model
        tmp_model = self.model.cpu()
        #self.valid_job.model = tmp_model
        del self.model
        if hasattr(self.valid_job, "model"):
            del self.valid_job.model
        gc.collect()
        if "cuda" in self.device:
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        self.parameter_client.barrier()
        num_eval_workers = self.config.get("job.distributed.num_eval_workers")
        if self.parameter_client.rank in range(self.min_rank, self.min_rank + num_eval_workers):
            # create a model for validation with entity embedder size
            #  batch_size x 2 + eval.chunk_size
            self.config.set(self.config.get("model") + ".create_eval", True)

            tmp_pretrain_model_filename = self.config.get("lookup_embedder.pretrain.model_filename")
            self.config.set("lookup_embedder.pretrain.model_filename", "")
            self.model = KgeModel.create(
                self.config, self.dataset, parameter_client=self.parameter_client
            )
            self.model.get_s_embedder().to_device(move_optim_data=False)
            self.model.get_p_embedder().to_device(move_optim_data=False)
            self.config.set("lookup_embedder.pretrain.model_filename", tmp_pretrain_model_filename)
            self.config.set(self.config.get("model") + ".create_eval", False)

            self.valid_job.model = self.model
            # validate and update learning rate
            super(TrainingJobNegativeSamplingDistributed, self).handle_validation(
                metric_name
            )

            # clean up valid model
            del self.model
            del self.valid_job.model
            gc.collect()
            if "cuda" in self.device:
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
        else:
            self.kge_lr_scheduler.step()
        self.parameter_client.barrier()
        self.model = tmp_model.to(self.device)
        del tmp_model
        gc.collect()

    def handle_running_checkpoint(self, checkpoint_every, checkpoint_keep):
        # since it is rather expensive to handle checkpoints in every epoch we only
        # do it every time we are evaluating now
        valid_every = self.config.get("valid.every")
        self.parameter_client.barrier()
        if self.parameter_client.rank == self.min_rank:
            self.save(self.config.checkpoint_file(self.epoch))
            delete_checkpoint_epoch = 0
            if checkpoint_every == 0:
                # do not keep any old checkpoints
                delete_checkpoint_epoch = self.epoch - valid_every
                # checkpoint every does not help a lot if we only store on valid
            elif checkpoint_keep > 0:
                # keep a maximum number of checkpoint_keep checkpoints
                delete_checkpoint_epoch = (
                    self.epoch - valid_every - valid_every * checkpoint_keep
                )
            if delete_checkpoint_epoch > 0:
                self._delete_checkpoint(
                    delete_checkpoint_epoch
                )
        self.parameter_client.barrier()

    def _init_dataloader(self):
        mp_context = (
            torch.multiprocessing.get_context("fork")
            if self.config.get("train.num_workers") > 0
            else None
        )
        self.loader = torch.utils.data.DataLoader(
            self.dataloader_dataset,
            sampler=InfiniteSequentialSampler(self.dataloader_dataset),
            collate_fn=self._get_collate_fun(),
            # shuffle needs to be False since it is handled in the dataset object
            shuffle=False,
            # batch size needs to be 1 since it is handled in the dataset object
            # batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
            multiprocessing_context=mp_context,
        )

    def run_epoch(self) -> Dict[str, Any]:
        """ Runs an epoch and returns its trace entry. """

        # create initial trace entry
        self.current_trace["epoch"] = dict(
            type=self.type_str,
            scope="epoch",
            epoch=self.epoch,
            split=self.train_split,
            size=self.num_examples,
        )
        if not self.is_forward_only:
            self.current_trace["epoch"].update(
                lr=[group["lr"] for group in self.optimizer.param_groups],
            )

        # run pre-epoch hooks (may modify trace)
        for f in self.pre_epoch_hooks:
            f(self)

        trace_entry = None
        local_partition_counter = -1
        while True:
            # variables that record various statitics
            sum_loss = 0.0
            sum_penalty = 0.0
            sum_penalties = defaultdict(lambda: 0.0)
            epoch_time = -time.time()
            prepare_time = 0.0
            forward_time = 0.0
            backward_time = 0.0
            optimizer_time = 0.0
            unique_time = 0.0
            pull_and_map_time = 0.0
            entity_pull_time = 0.0
            relation_pull_time = 0.0
            pre_pull_time = 0.0
            cpu_gpu_time = 0.0
            ps_wait_time = 0.0
            ps_set_time = 0.0
            dataloader_time = 0.0
            scheduler_time = -time.time()

            # load new work package
            work, work_entities, work_relations = self.work_scheduler_client.get_work()
            self.entity_partition_localized = False
            self.relation_partition_localized = False
            if work is None:
                break
            local_partition_counter += 1
            self.work_pre_localized = False
            if work_entities is not None and self.entity_localize:
                self.model.get_s_embedder().localize(work_entities)
                self.entity_partition_localized = True
            if work_relations is not None and self.relation_localize:
                self.model.get_p_embedder().localize(work_relations)
                self.relation_partition_localized = True
            if self.entity_sync_level == "partition":
                if work_entities is not None:
                    entity_pull_time -= time.time()
                    actual_entity_pull_time, entity_cpu_gpu_time = self.model.get_s_embedder()._pull_embeddings(work_entities)
                    self.model.get_s_embedder().global_to_local_mapper[
                        work_entities
                    ] = torch.arange(len(work_entities), dtype=torch.long, device="cpu")
                    entity_pull_time += time.time()
                    cpu_gpu_time += entity_cpu_gpu_time
                else:
                    raise ValueError(
                        "the used work-scheduler seems not to support "
                        "syncing entities on a partition level"
                    )
            if self.relation_sync_level == "partition":
                if work_relations is not None:
                    relation_pull_time -= time.time()
                    actual_relation_pull_time, relation_cpu_gpu_time = self.model.get_p_embedder()._pull_embeddings(work_relations)
                    self.model.get_p_embedder().global_to_local_mapper[
                        work_relations
                    ] = torch.arange(
                        len(work_relations), dtype=torch.long, device="cpu"
                    )
                    relation_pull_time += time.time()
                    cpu_gpu_time += relation_cpu_gpu_time
                else:
                    raise ValueError(
                        "the used work-scheduler seems not to support "
                        "syncing relations on a partition level"
                    )

            if (
                work_entities is not None
                and self.config.get("negative_sampling.sampling_type") == "pooled"
            ):
                self.local_entities = work_entities
                self._sampler.set_pool(work_entities, S)
                self._sampler.set_pool(work_entities, O)
            self.dataloader_dataset.set_samples(
                work, self.epoch, local_partition_counter
            )
            if self.loader is None:
                self._init_dataloader()
                self.iter_dataloader = iter(self.loader)
            object.__setattr__(self.loader, "sampler",
                               InfiniteSequentialSampler(self.dataloader_dataset))
            object.__setattr__(self.iter_dataloader, "_sampler_iter",
                               iter(self.iter_dataloader._index_sampler))
            scheduler_time += time.time()

            # process each batch
            pre_load_batches = deque()
            batch_index = 0
            num_prepulls = max(self.entity_pre_pull, self.relation_pre_pull, self.pre_localize_batch, 1)
            while batch_index < len(self.dataloader_dataset):
                while len(pre_load_batches) < num_prepulls + 1:
                    prepare_time -= time.time()
                    dataloader_time -= time.time()
                    next_batch = next(self.iter_dataloader)
                    while next_batch["epoch"] != self.epoch or next_batch[
                        "local_partition_id"] != local_partition_counter:
                        next_batch = next(self.iter_dataloader)
                    next_batch = self._map_ids_to_local(next_batch)
                    pre_load_batches.append(next_batch)
                    dataloader_time += time.time()
                    pre_pull_time -= time.time()
                    if next_batch is not None:
                        self._prepare_batch_ahead(pre_load_batches)
                    pre_pull_time += time.time()
                    prepare_time += time.time()
                batch = pre_load_batches.popleft()

                # create initial batch trace (yet incomplete)
                self.current_trace["batch"] = {
                    "type": self.type_str,
                    "scope": "batch",
                    "epoch": self.epoch,
                    "split": self.train_split,
                    "batch": batch_index,
                    "batches": len(self.loader),
                }
                if not self.is_forward_only:
                    self.current_trace["batch"].update(
                        lr=[group["lr"] for group in self.optimizer.param_groups],
                    )

                # run the pre-batch hooks (may update the trace)
                for f in self.pre_batch_hooks:
                    f(self, batch_index)

                # process batch (preprocessing + forward pass + backward pass on loss)
                batch_result: TrainingJob._ProcessBatchResult = self._auto_subbatched_process_batch(
                    batch_index, batch
                )
                sum_loss += batch_result.avg_loss * batch_result.size

                # determine penalty terms (forward pass)
                batch_forward_time = batch_result.forward_time - time.time()
                penalties_torch = self.model.penalty(
                    epoch=self.epoch,
                    batch_index=batch_index,
                    num_batches=self.dataloader_dataset.get_real_len(),
                    batch=batch,
                )
                batch_forward_time += time.time()

                # backward pass on penalties
                batch_backward_time = batch_result.backward_time - time.time()
                penalty = 0.0
                for index, (penalty_key, penalty_value_torch) in enumerate(
                    penalties_torch
                ):
                    if not self.is_forward_only:
                        penalty_value_torch.backward()
                    penalty += penalty_value_torch.item()
                    sum_penalties[penalty_key] += penalty_value_torch.item()
                sum_penalty += penalty
                batch_backward_time += time.time()

                # determine full cost
                cost_value = batch_result.avg_loss + penalty

                # abort on nan
                if self.abort_on_nan and math.isnan(cost_value):
                    raise FloatingPointError("Cost became nan, aborting training job")

                # print memory stats
                if self.epoch == 1 and batch_index == 0:
                    if self.device.startswith("cuda"):
                        with torch.cuda.device(self.device):
                            self.config.log(
                                "CUDA memory after first batch: allocated={:14,} "
                                "reserved={:14,} max_allocated={:14,}".format(
                                    torch.cuda.memory_allocated(self.device),
                                    torch.cuda.memory_reserved(self.device),
                                    torch.cuda.max_memory_allocated(self.device),
                                )
                            )

                # update parameters
                batch_optimizer_time = -time.time()
                if not self.is_forward_only:
                    self.optimizer.step()
                batch_optimizer_time += time.time()

                if self.entity_sync_level == "batch":
                    self.model.get_s_embedder().push_back()
                if self.relation_sync_level == "batch":
                    self.model.get_p_embedder().push_back()

                # update batch trace with the results
                self.current_trace["batch"].update(
                    {
                        "size": batch_result.size,
                        "avg_loss": batch_result.avg_loss,
                        # "penalties": [p.item() for k, p in penalties_torch],
                        "penalty": penalty,
                        "cost": cost_value,
                        "prepare_time": batch_result.prepare_time,
                        "forward_time": batch_forward_time,
                        "backward_time": batch_backward_time,
                        "optimizer_time": batch_optimizer_time,
                        "event": "batch_completed",
                    }
                )

                # run the post-batch hooks (may modify the trace)
                for f in self.post_batch_hooks:
                    f(self)

                # output, then clear trace
                if self.trace_batch:
                    self.trace(**self.current_trace["batch"])
                self.current_trace["batch"] = None

                # print console feedback
                self.config.print(
                    (
                        "\r"  # go back
                        + "{}  batch{: "
                        + str(1 + int(math.ceil(math.log10(self.dataloader_dataset.get_real_len()))))
                        + "d}/{}"
                        + ", avg_loss {:.4E}, penalty {:.4E}, cost {:.4E}, time {:6.2f}s"
                        + "\033[K"  # clear to right
                    ).format(
                        self.config.log_prefix,
                        batch_index,
                        self.dataloader_dataset.get_real_len() - 1,
                        batch_result.avg_loss,
                        penalty,
                        cost_value,
                        batch_result.prepare_time
                        + batch_forward_time
                        + batch_backward_time
                        + batch_optimizer_time,
                    ),
                    end="",
                    flush=True,
                )

                # update epoch times
                prepare_time += batch_result.prepare_time
                forward_time += batch_forward_time
                backward_time += batch_backward_time
                optimizer_time += batch_optimizer_time
                pull_and_map_time += batch_result.pull_and_map_time
                entity_pull_time += batch_result.entity_pull_time
                relation_pull_time += batch_result.relation_pull_time
                unique_time += batch_result.unique_time
                cpu_gpu_time += batch_result.cpu_gpu_time
                ps_wait_time += batch_result.ps_wait_time

                batch_index += 1

            # all done; now trace and log
            epoch_time += time.time()
            self.config.print("\033[2K\r", end="", flush=True)  # clear line and go back

            other_time = (
                epoch_time
                - prepare_time
                - forward_time
                - backward_time
                - optimizer_time
                - scheduler_time
            )

            if self.entity_sync_level == "partition":
                ps_set_time -= time.time()
                self.model.get_s_embedder().set_embeddings()
                ps_set_time += time.time()
                # this is expensive and unnecessary
                # self.model.get_s_embedder().global_to_local_mapper[:] = -1
                self.model.get_s_embedder().push_back()
            if self.relation_sync_level == "partition":
                ps_set_time -= time.time()
                self.model.get_p_embedder().set_embeddings()
                ps_set_time += time.time()
                # self.model.get_p_embedder().global_to_local_mapper[:] = -1
                self.model.get_p_embedder().push_back()
            self.work_scheduler_client.work_done()

            # add results to trace entry
            self.current_trace["epoch"].update(
                dict(
                    avg_loss=sum_loss / self.num_examples,
                    avg_penalty=sum_penalty / self.dataloader_dataset.get_real_len(),
                    avg_penalties={
                        k: p / self.dataloader_dataset.get_real_len() for k, p in sum_penalties.items()
                    },
                    avg_cost=sum_loss / self.num_examples
                    + sum_penalty / self.dataloader_dataset.get_real_len(),
                    epoch_time=epoch_time,
                    prepare_time=prepare_time,
                    ps_wait_time=ps_wait_time,
                    unique_time=unique_time,
                    pull_and_map_time=pull_and_map_time,
                    pre_pull_time=pre_pull_time,
                    entity_pull_time=entity_pull_time,
                    relation_pull_time=relation_pull_time,
                    ps_set_time=ps_set_time,
                    cpu_gpu_time=cpu_gpu_time,
                    forward_time=forward_time,
                    backward_time=backward_time,
                    optimizer_time=optimizer_time,
                    scheduler_time=scheduler_time,
                    other_time=other_time,
                    embedding_mapping_time=self.model.get_s_embedder().mapping_time
                    + self.model.get_p_embedder().mapping_time,
                    event="epoch_completed",
                    batches=len(self.loader),
                    dataloader_time=dataloader_time,
                )
            )
            self.model.get_p_embedder().mapping_time = 0.0
            self.model.get_s_embedder().mapping_time = 0.0

            # run hooks (may modify trace)
            for f in self.post_epoch_hooks:
                f(self)

            # output the trace, then clear it
            trace_entry = self.trace(
                **self.current_trace["epoch"], echo=False, log=True
            )
        self.config.log(
            format_trace_entry("train_epoch", trace_entry, self.config), prefix="  "
        )
        self.current_trace["epoch"] = None
        return trace_entry

    def _delete_checkpoint(self, checkpoint_id):
        filename = self.config.checkpoint_file(checkpoint_id)
        super(TrainingJobNegativeSamplingDistributed, self)._delete_checkpoint(
            checkpoint_id
        )
        file, file_ending = filename.rsplit(".", 1)
        if os.path.exists(f"{file}_entities"):
            shutil.rmtree(f"{file}_entities")
        if os.path.exists(f"{file}_relations.{file_ending}"):
            os.remove(f"{file}_relations.{file_ending}")

    def save(self, filename) -> None:
        if self.parameter_client.rank == get_min_rank(self.config):
            # todo: we do not need to store the weights of the emebdders and optim here
            super(TrainingJobNegativeSamplingDistributed, self).save(filename)
            local_model_size = self.model.get_s_embedder().vocab_size
            global_model_size = self.model.get_s_embedder().complete_vocab_size
            entity_dim = self.model.get_s_embedder().dim
            optimizer_dim = self.model.get_s_embedder().optimizer_dim
            chunk_size = min(max(1000000, local_model_size), global_model_size)
            empty_pull_tensor = torch.empty(
                [chunk_size, entity_dim + optimizer_dim], device="cpu"
            )
            num_entities = self.dataset.num_entities()
            file, file_ending = filename.rsplit(".", 1)
            entities_dir = f"{file}_entities"
            if not os.path.exists(entities_dir):
                os.mkdir(entities_dir)
            for chunk_number in range(math.ceil(num_entities / chunk_size)):
                chunk_start = chunk_size * chunk_number
                chunk_end = min(chunk_size * (chunk_number + 1), num_entities)
                entity_ids = torch.arange(chunk_start, chunk_end, dtype=torch.long)
                lapse_offset = self.model.get_s_embedder().lapse_offset
                pull_tensor = empty_pull_tensor[: len(entity_ids)]
                self.parameter_client.pull(entity_ids + lapse_offset, pull_tensor)
                torch.save(
                    pull_tensor,
                    os.path.join(
                        entities_dir, f"{chunk_start}-{chunk_end}.{file_ending}"
                    ),
                )
            lapse_offset = self.model.get_p_embedder().lapse_offset
            pull_tensor = self.model.get_p_embedder().pull_tensors[0][1]
            relation_ids = torch.arange(self.dataset.num_relations(), dtype=torch.long)
            self.parameter_client.pull(relation_ids + lapse_offset, pull_tensor)
            torch.save(pull_tensor, f"{file}_relations.{file_ending}")

