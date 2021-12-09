import torch
try:
    import lapse
except ImportError:
    pass
from enum import IntEnum
from torch import distributed as dist

from kge.distributed.misc import get_min_rank, get_optimizer_dim, initialize_worker_groups, set_dmlc_environment, \
    set_master_environment


class TORCH_PARAMETER_SERVER_CMDS(IntEnum):
    PULL_CMD = 0
    PUSH_CMD = 1
    SET_CMD = 2
    GET_ENTITY_LR_CMD = 3
    GET_RELATION_LR_CMD = 4
    SET_ENTITY_LR_CMD = 5
    SET_RELATION_LR_CMD = 6
    GET_OPTIM_STEP_CMD = 7
    STEP_OPTIM_CMD = 8
    BARRIER_CMD = 9
    SHUTDOWN_CMD = 10


class KgeParameterServer:
    @staticmethod
    def get_parameter_server():
        raise NotImplementedError()


class LapseParameterServer:
    @staticmethod
    def get_parameter_server(config, num_keys):
        """In Lapse we have a server for every worker, therefore we don't use a lock"""
        set_dmlc_environment(config, role="server")

        embedding_dim = config.get("lookup_embedder.dim")
        optimizer_dim = get_optimizer_dim(config, embedding_dim)
        num_workers_per_server = 1
        lapse.setup(num_keys, num_workers_per_server)
        return lapse.Server(num_keys, embedding_dim + optimizer_dim)


class TorchParameterServer:
    def __init__(self, world_size: int, num_keys: int, dim: int):
        self.rank = 0
        self.num_clients = world_size - 2
        self.dim = dim
        self.data_type = torch.float32
        self.data = torch.zeros((num_keys, dim), dtype=self.data_type)
        self.entity_lr = 0
        self.relation_lr = 0
        self.entity_optim_step = 0
        self.relation_optim_step = 0
        self.start()

    def start(self):
        barrier_count = 0
        shutdown_count = 0
        lr_buffer = torch.zeros(1, dtype=torch.float32)
        while True:
            # cmd_buffer consists of cmd_number, key_len
            cmd_buffer = torch.full((2,), -1, dtype=torch.long)
            rank = dist.recv(cmd_buffer)
            cmd = cmd_buffer[0].item()
            if cmd == TORCH_PARAMETER_SERVER_CMDS.PULL_CMD:
                key_len = cmd_buffer[1].item()
                keys = self._receive_keys(rank, key_len)
                data = self.data[keys, :]
                dist.send(data, dst=rank)
            if cmd == TORCH_PARAMETER_SERVER_CMDS.PUSH_CMD:
                key_len = cmd_buffer[1].item()
                keys = self._receive_keys(rank, key_len)
                self._handle_push(rank, keys)
            if cmd == TORCH_PARAMETER_SERVER_CMDS.SET_CMD:
                key_len = cmd_buffer[1].item()
                keys = self._receive_keys(rank, key_len)
                self._handle_set(rank, keys)
            if cmd == TORCH_PARAMETER_SERVER_CMDS.GET_ENTITY_LR_CMD:
                lr_buffer[0] = self.entity_lr
                dist.send(lr_buffer, rank)
            if cmd == TORCH_PARAMETER_SERVER_CMDS.GET_RELATION_LR_CMD:
                lr_buffer[0] = self.relation_lr
                dist.send(lr_buffer, rank)
            if cmd == TORCH_PARAMETER_SERVER_CMDS.SET_ENTITY_LR_CMD:
                dist.recv(lr_buffer, src=rank)
                self.entity_lr = lr_buffer.item()
            if cmd == TORCH_PARAMETER_SERVER_CMDS.SET_RELATION_LR_CMD:
                dist.recv(lr_buffer, src=rank)
                self.relation_lr = lr_buffer.item()
            if cmd == TORCH_PARAMETER_SERVER_CMDS.GET_OPTIM_STEP_CMD:
                parameter_index = cmd_buffer[1].item()
                if parameter_index == 0:
                    cmd_buffer[1] = self.entity_optim_step
                elif parameter_index == 1:
                    cmd_buffer[1] = self.relation_optim_step
                dist.send(cmd_buffer, rank)
            if cmd == TORCH_PARAMETER_SERVER_CMDS.STEP_OPTIM_CMD:
                parameter_index = cmd_buffer[1].item()
                if parameter_index == 0:
                    self.entity_optim_step += 1
                elif parameter_index == 1:
                    self.relation_optim_step += 1
            if cmd == TORCH_PARAMETER_SERVER_CMDS.BARRIER_CMD:
                barrier_count += 1
                if barrier_count == self.num_clients:
                    barrier_count = 0
                    dist.barrier()
            if cmd == TORCH_PARAMETER_SERVER_CMDS.SHUTDOWN_CMD:
                shutdown_count += 1
                if shutdown_count == self.num_clients:
                    print("shutting down parameter server")
                    break

    @staticmethod
    def _receive_keys(rank, key_len):
        keys = torch.empty((key_len,), dtype=torch.long)
        dist.recv(keys, src=rank)
        return keys

    def _handle_push(self, rank, keys):
        push_data = torch.empty((len(keys), self.dim), dtype=self.data_type)
        dist.recv(push_data, src=rank)
        self.data[keys, :] += push_data

    def _handle_set(self, rank, keys):
        set_data = torch.empty((len(keys), self.dim), dtype=self.data_type)
        dist.recv(set_data, src=rank)
        self.data[keys, :] = set_data


def init_lapse_scheduler(config, num_keys):
    # we are only initializing dist here to have the same ranks for lapse and torch
    set_master_environment(config)
    set_dmlc_environment(config, role="scheduler")
    num_workers_per_server = 1
    lapse.scheduler(num_keys, num_workers_per_server)


def init_torch_server(config, num_keys):
    num_clients = config.get("job.distributed.num_workers")
    min_rank = get_min_rank(config)
    world_size = num_clients + min_rank
    dim = config.get("lookup_embedder.dim")
    optimizer_dim = get_optimizer_dim(config, dim)
    set_master_environment(config)
    # process groups need to be initialized in every process
    initialize_worker_groups(config, 0)

    TorchParameterServer(world_size, num_keys, dim + optimizer_dim)
