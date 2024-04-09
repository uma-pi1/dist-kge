import math
import time
from torch import Tensor
import torch.nn
import torch.nn.functional

import torch
from collections import deque

from kge import Config, Dataset
from kge.model import LookupEmbedder, KgeEmbedder
from kge.distributed.misc import get_optimizer_dim

from typing import List


class DistributedLookupEmbedder(LookupEmbedder):
    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key: str,
        vocab_size: int,
        parameter_client: "KgeParameterClient",
        complete_vocab_size,
        lapse_offset=0,
        init_for_load_only=False,
    ):
        super().__init__(
            config,
            dataset,
            configuration_key,
            vocab_size,
            init_for_load_only=init_for_load_only,
        )
        self.optimizer_dim = get_optimizer_dim(config, self.dim)
        self.optimizer_values = torch.zeros(
            (self.vocab_size, self.optimizer_dim),
            dtype=torch.float32,
            requires_grad=False,
        )

        self.complete_vocab_size = complete_vocab_size
        self.parameter_client = parameter_client
        self.lapse_offset = lapse_offset
        self.pulled_ids = None
        # global to local mapper only used in sync level partition
        self.global_to_local_mapper = torch.full(
            (self.dataset.num_entities(),), -1, dtype=torch.long, device="cpu"
        )

        # maps the local embeddings to the embeddings in lapse
        # used in optimizer
        self.local_to_lapse_mapper = torch.full(
            (vocab_size,), -1, dtype=torch.long, requires_grad=False
        )
        self.pull_dim = self.dim + self.optimizer_dim
        self.unnecessary_dim = self.parameter_client.dim - self.pull_dim

        # 3 pull tensors to pre-pull up to 3 batches
        # first boolean denotes if the tensor is free
        number_of_pre_pulls = 0
        if "entity" in self.configuration_key:
            number_of_pre_pulls = self.config.get("job.distributed.entity_pre_pull")
        elif "relation" in self.configuration_key:
            number_of_pre_pulls = self.config.get("job.distributed.relation_pre_pull")
        self.pull_tensors = []
        for i in range(number_of_pre_pulls + 1):
            self.pull_tensors.append(
                [
                    True,
                    torch.empty(
                        (self.vocab_size, self.parameter_client.dim),
                        # (self.vocab_size, self.dim + self.optimizer_dim),
                        dtype=torch.float32,
                        device="cpu",
                        requires_grad=False,
                    ),
                ]
            )

        if "cuda" in config.get("job.device"):
            # only pin tensors if we are using gpu
            # otherwise gpu memory will be allocated for no reason
            with torch.cuda.device(config.get("job.device")):
                for i in range(len(self.pull_tensors)):
                    self.pull_tensors[i][1] = self.pull_tensors[i][1].pin_memory()

        self.num_pulled = 0
        self.mapping_time = 0.0
        # self.pre_pulled = None
        self.pre_pulled = deque()

    def to_device(self, move_optim_data=True):
        """Needs to be called after model.to(self.device)"""
        if move_optim_data:
            self.optimizer_values = self.optimizer_values.to(
                self._embeddings.weight.device
            )

    @torch.no_grad()
    def init_pretrained(self, pretrained_embedder: KgeEmbedder) -> None:
        (
            self_intersect_ind,
            pretrained_intersect_ind,
        ) = self._intersect_ids_with_pretrained_embedder(pretrained_embedder)
        # process in chunks to reduce memory footprint
        chunk_size = 1000000
        num_objects = len(pretrained_intersect_ind)
        for chunk_number in range(math.ceil(num_objects / chunk_size)):
            chunk_start = chunk_size * chunk_number
            chunk_end = min(chunk_size * (chunk_number + 1), num_objects)
            current_chunk_size = chunk_end - chunk_start
            pretrained_embeddings = pretrained_embedder.embed(
                torch.from_numpy(pretrained_intersect_ind[chunk_start:chunk_end])
            )
            self.parameter_client.push(
                torch.from_numpy(self_intersect_ind[chunk_start:chunk_end]) + self.lapse_offset,
                torch.cat(
                    (
                        pretrained_embeddings,
                        torch.zeros(
                            (current_chunk_size, self.optimizer_dim + self.unnecessary_dim),
                            dtype=pretrained_embeddings.dtype,
                        ),
                    ),
                    dim=1,
                ),
            )

    def push_all(self):
        if self.unnecessary_dim > 0:
            # todo: this is currently just a workaround until we support parameter
            #  of different lengths
            push_tensor = torch.cat(
                (
                    self._embeddings.weight.detach().cpu(),
                    self.optimizer_values.cpu(),
                    torch.empty(
                        [len(self.optimizer_values), self.unnecessary_dim],
                        device="cpu",
                        dtype=self.optimizer_values.dtype,
                    ),
                ),
                dim=1,
            )
        else:
            push_tensor = torch.cat(
                (self._embeddings.weight.detach().cpu(), self.optimizer_values.cpu()),
                dim=1,
            )
        self.parameter_client.push(
            torch.arange(self.vocab_size) + self.lapse_offset, push_tensor
        )

    def pull_all(self):
        self._pull_embeddings(torch.arange(self.complete_vocab_size))

    def set_embeddings(self):
        # storing set_indexes and set_tensors in self to keep them alive until async
        #  set is finished
        self.set_indexes = self.pulled_ids + self.lapse_offset
        num_pulled = len(self.set_indexes)
        # move tensors to cpu before cat to reduce gpu memory usage
        if self.unnecessary_dim > 0:
            self.set_tensor = torch.cat(
                (
                    self._embeddings.weight[:num_pulled].detach().cpu(),
                    self.optimizer_values[:num_pulled].cpu(),
                    torch.empty((num_pulled, self.unnecessary_dim), device="cpu"),
                ),
                dim=1,
            )
        else:
            self.set_tensor = torch.cat(
                (
                    self._embeddings.weight[:num_pulled].detach().cpu(),
                    self.optimizer_values[:num_pulled].cpu(),
                ),
                dim=1,
            )
        self.parameter_client.set(self.set_indexes, self.set_tensor, asynchronous=True)

    def _get_free_pull_tensor(self):
        for i, (free, pull_tensor) in enumerate(self.pull_tensors):
            if free:
                self.pull_tensors[i][0] = False
                return i, pull_tensor

    @torch.no_grad()
    def pre_pull(self, indexes):
        pull_indexes = (indexes + self.lapse_offset).cpu()
        pull_tensor_index, pull_tensor = self._get_free_pull_tensor()
        pull_tensor = pull_tensor[: len(indexes)]
        pull_future = self.parameter_client.pull(
            pull_indexes, pull_tensor, asynchronous=True
        )
        self.pre_pulled.append(
            {
                "indexes": indexes,
                "pull_indexes": pull_indexes,
                "pull_tensor": pull_tensor,
                "pull_future": pull_future,
                "pull_tensor_index": pull_tensor_index,
            }
        )

    def pre_pulled_to_device(self):
        if len(self.pre_pulled) > 2:
            # id 0 is from the batch currently processed
            # last one is the one pulled from ps
            # we are moving the second last
            self.parameter_client.wait(self.pre_pulled[-2]["pull_future"])
            self.pre_pulled[-2]["pull_tensor"] = self.pre_pulled[-2]["pull_tensor"].to(
                self._embeddings.weight.device, non_blocking=True
            )

    @torch.no_grad()
    def _pull_embeddings(self, indexes):
        cpu_gpu_time = 0.0
        pull_time = 0.0
        device = self._embeddings.weight.device
        len_indexes = len(indexes)
        if len(self.pre_pulled) > 0:
            # todo: add workaround for relations here as well
            # todo: clean up this method
            pre_pulled = self.pre_pulled.popleft()
            self.pulled_ids = pre_pulled["indexes"]
            self.parameter_client.wait(pre_pulled["pull_future"])
            self.local_to_lapse_mapper[:len_indexes] = pre_pulled["pull_indexes"]
            cpu_gpu_time -= time.time()
            pre_pulled_tensor = pre_pulled["pull_tensor"].to(device)
            cpu_gpu_time += time.time()
            pulled_embeddings, pulled_optim_values = torch.split(
                pre_pulled_tensor, [self.dim, self.optimizer_dim], dim=1
            )
            self._embeddings.weight[:len_indexes] = pulled_embeddings
            self.optimizer_values[:len_indexes] = pulled_optim_values
            self.pull_tensors[pre_pulled["pull_tensor_index"]][0] = True
            return pull_time, cpu_gpu_time

        self.pulled_ids = indexes
        pull_indexes = (indexes + self.lapse_offset).cpu()
        self.local_to_lapse_mapper[:len_indexes] = pull_indexes
        pull_tensor = self.pull_tensors[0][1][:len_indexes]
        pull_time -= time.time()
        self.parameter_client.pull(pull_indexes, pull_tensor)
        pull_time += time.time()
        cpu_gpu_time -= time.time()
        # split tensor already before moving to gpu to reduce memory footprint on gpu
        pulled_embeddings, pulled_optim_values, _ = torch.split(
            pull_tensor, [self.dim, self.optimizer_dim, self.unnecessary_dim], dim=1
        )
        self._embeddings.weight.data[:len_indexes].copy_(pulled_embeddings, non_blocking=True)
        self.optimizer_values[:len_indexes].copy_(pulled_optim_values, non_blocking=True)
        cpu_gpu_time += time.time()
        return pull_time, cpu_gpu_time

    def localize(self, indexes: Tensor, asynchronous=False, make_unique=False):
        if make_unique:
            indexes = torch.unique(indexes)
        self.parameter_client.localize(
            (indexes + self.lapse_offset).cpu(), asynchronous
        )

    def _embed(self, indexes: Tensor) -> Tensor:
        long_indexes = indexes.long()
        return self._embeddings(long_indexes)

    def embed(self, indexes: Tensor) -> Tensor:
        long_indexes = indexes.long()
        return self._postprocess(self._embeddings(long_indexes))

    def embed_all(self) -> Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def push_back(self):
        self.local_to_lapse_mapper[:] = -1
        self.num_pulled = 0

    def _embeddings_all(self) -> Tensor:
        # TODO: this should not be possible in the distributed lookup embedder
        raise NotImplementedError

    def penalty(self, **kwargs) -> List[Tensor]:
        # TODO factor out to a utility method
        # Avoid calling lookup embedder penalty and instead call KgeEmbedder penalty
        result = KgeEmbedder.penalty(self, **kwargs)
        if self.regularize == "" or self.get_option("regularize_weight") == 0.0:
            pass
        elif self.regularize == "lp":
            p = (
                self.get_option("regularize_args.p")
                if self.has_option("regularize_args.p")
                else 2
            )
            regularize_weight = self._get_regularize_weight()
            if not self.get_option("regularize_args.weighted"):
                # unweighted Lp regularization
                parameters = self._embeddings_all()
                result += [
                    (
                        f"{self.configuration_key}.L{p}_penalty",
                        (regularize_weight / p * parameters.norm(p=p) ** p).sum(),
                    )
                ]
            else:
                # weighted Lp regularization
                unique_indexes, counts = torch.unique(
                    kwargs["indexes"], return_counts=True
                )
                parameters = self._embed(unique_indexes)
                if p % 2 == 1:
                    parameters = torch.abs(parameters)
                result += [
                    (
                        f"{self.configuration_key}.L{p}_penalty",
                        (
                            regularize_weight
                            / p
                            * (parameters ** p * counts.float().view(-1, 1))
                        ).sum()
                        # In contrast to unweighted Lp regularization, rescaling by
                        # number of triples/indexes is necessary here so that penalty
                        # term is correct in expectation
                        / len(kwargs["indexes"]),
                    )
                ]
        else:  # unknown regularization
            raise ValueError(f"Invalid value regularize={self.regularize}")

        return result
