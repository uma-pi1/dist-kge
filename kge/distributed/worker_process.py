import os
import gc
import datetime
from typing import Optional
from copy import deepcopy
import torch
from torch import multiprocessing as mp

from kge.misc import set_seeds
from kge.job import Job
from .parameter_client import KgeParameterClient
from .parameter_server import LapseParameterServer
from .misc import get_min_rank, get_num_keys, get_optimizer_dim, set_master_environment

class WorkerProcessPool:
    """
    Creates all the train-workers for distributed training
    """
    def __init__(
        self,
        config,
        dataset,
        checkpoint_name: Optional[str] = None,
    ):
        num_workers_machine = config.get("job.distributed.num_workers_machine")
        already_init_workers = config.get("job.distributed.already_init_workers")

        config.log(f"creating worker process pool with {config.get('job.distributed.num_workers')} workers")
        self.workers = []
        configs = {}
        parameters = None
        if config.get("job.distributed.parameter_server") == "shared":
            # When we use a shared tensor as a parameter server, the shared memory needs
            # to be allocated before creating the worker processes and shared between
            # workers
            num_keys = get_num_keys(config, dataset)
            embedding_dim = config.get("lookup_embedder.dim")
            optimizer_dim = get_optimizer_dim(config, embedding_dim)
            parameters = torch.empty(
                (num_keys, embedding_dim + optimizer_dim),
                dtype=torch.float32,
                requires_grad=False
            ).share_memory_()
        for rank in range(num_workers_machine):
            if rank == 0:
                self.recv_end, send_end = mp.Pipe(False)
            else:
                send_end = None
            configs[rank] = deepcopy(config)
            configs[rank].init_folder()
            worker = WorkerProcess(
                rank + already_init_workers,
                configs[rank],
                dataset,
                parameters=parameters,
                checkpoint_name=checkpoint_name,
                result_pipe=send_end
            )
            worker.start()
            self.workers.append(worker)

    def join(self):
        """Wait for all workers"""
        try:
            valid_trace = self.recv_end.recv()
        except:
            valid_trace = None
        for worker in self.workers:
            worker.join()
        return valid_trace

    def terminate(self):
        print("terminating worker process pool")
        for worker in self.workers:
            worker.terminate()

    def kill(self):
        print("killing worker process pool")
        for worker in self.workers:
            worker.kill()


class WorkerProcess(mp.get_context("spawn").Process):
    """Train worker"""
    def __init__(
        self,
        rank,
        config,
        dataset,
        parameters=None,
        checkpoint_name: Optional[str] = None,
        result_pipe=None,
    ):
        # rank = rank + 1
        daemon = config.get("train.num_workers") <= 0 and config.get("eval.num_workers") <= 0
        super().__init__(daemon=daemon, name=f"Worker #{rank}")
        self.rank = rank
        self.num_keys = get_num_keys(config, dataset)
        self.config = config
        self.dataset = dataset
        self.parameters = parameters
        self.checkpoint_name = checkpoint_name
        self.result_pipe = result_pipe

    def run(self):
        torch_device = self.config.get("job.device")
        if self.config.get("job.device") == "cuda":
            torch_device = "cuda:0"
        if torch_device != "cpu":
            torch.cuda.set_device(torch_device)
        # seeds need to be set in every process
        set_seeds(self.config, self.rank)

        set_master_environment(self.config)
        min_rank = get_min_rank(self.config)
        print("before init", self.rank + min_rank)

        # create parameter server
        server = None
        if self.config.get("job.distributed.parameter_server") == "lapse":
            server = LapseParameterServer.get_parameter_server(self.config, self.num_keys)
        elif self.config.get("job.distributed.parameter_server") == "shared":
            server = self.parameters

        # create train-worker config, dataset and folder
        device_pool: list = self.config.get("job.device_pool")
        if len(device_pool) == 0:
            device_pool.append(self.config.get("job.device"))
        config = deepcopy(self.config)
        config.set("job.device", device_pool[self.rank % len(device_pool)])
        config.folder = os.path.join(self.config.folder, f"worker-{self.rank}")
        config.init_folder()

        parameter_client = KgeParameterClient.create(
            config=config,
            server_id=0,
            client_id=self.rank + min_rank,
            server=server,
            num_keys=self.num_keys,
        )
        # don't re-initialize the model after loading checkpoint
        init_for_load_only = self.checkpoint_name is not None
        job = Job.create(
            config=config,
            dataset=self.dataset,
            parameter_client=parameter_client,
            init_for_load_only=init_for_load_only,
        )
        if self.checkpoint_name is not None:
            job.load_distributed(checkpoint_name=self.checkpoint_name)

        job.run()

        # all done, clean up
        print("shut down everything")
        parameter_client.barrier()
        if hasattr(job, "work_scheduler_client"):
            job.work_scheduler_client.shutdown()
        parameter_client.shutdown()
        # delete all occurrences of the parameter client to properly shutdown lapse
        # del job
        del job.parameter_client
        del job.model.get_s_embedder().parameter_client
        del job.model.get_p_embedder().parameter_client
        del job.model
        if hasattr(job, "optimizer"):
            del job.optimizer
        del parameter_client
        gc.collect()  # make sure lapse-worker destructor is called
        # shutdown server
        if server is not None and type(server) != torch.Tensor:
            server.shutdown()
        if self.result_pipe is not None:
            if hasattr(job, "valid_trace"):
                # if we valid from checkpoint there is no valid trace
                self.result_pipe.send(job.valid_trace)
            else:
                self.result_pipe.send(None)
