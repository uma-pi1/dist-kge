import os
import gc
import datetime
import torch  # import torch before lapse
try:
    import lapse
except ImportError:
    pass
from typing import Optional, Dict
from copy import deepcopy
from torch import multiprocessing as mp
from torch import distributed as dist

from kge.misc import set_seeds
from kge.job import Job
from kge.util.io import load_checkpoint
from .parameter_client import KgeParameterClient
from .misc import get_min_rank


class WorkerProcessPool:
    """
    Creates all the train-workers for distributed training
    """
    def __init__(
        self,
        num_total_workers,
        num_workers_machine,
        already_init_workers,
        num_keys,
        num_meta_keys,
        embedding_dim,
        optimizer_dim,
        config,
        dataset,
        checkpoint_name: Optional[str] = None,
    ):
        config.log(f"creating worker process pool with {num_total_workers} worker")
        self.workers = []
        configs = {}
        parameters=None
        if config.get("job.distributed.parameter_server") == "shared":
            parameters = torch.empty((num_keys, embedding_dim + optimizer_dim), dtype=torch.float32, requires_grad=False).share_memory_()
        for rank in range(num_workers_machine):
            if rank == 0:
                self.recv_end, send_end = mp.Pipe(False)
            else:
                send_end = None
            configs[rank] = deepcopy(config)
            #configs[rank].set(config.get("model") + ".create_complete", False)
            configs[rank].init_folder()
            worker = WorkerProcess(
                rank + already_init_workers,
                num_total_workers,
                num_keys,
                num_meta_keys,
                embedding_dim,
                optimizer_dim,
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
        num_total_workers,
        num_keys,
        num_meta_keys,
        embedding_dim,
        optimizer_dim,
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
        self.num_total_workers = num_total_workers
        self.num_keys = num_keys
        self.num_meta_keys = num_meta_keys
        self.embedding_dim = embedding_dim
        self.optimizer_dim = optimizer_dim
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

        os.environ["MASTER_ADDR"] = self.config.get("job.distributed.master_ip")
        os.environ["MASTER_PORT"] = str(self.config.get("job.distributed.master_port"))
        min_rank = get_min_rank(self.config)
        print("before init", self.rank + min_rank)
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            world_size=self.num_total_workers + min_rank,
            rank=self.rank + min_rank,
            timeout=datetime.timedelta(hours=6),
        )
        worker_ranks = list(range(min_rank, self.num_total_workers+min_rank))
        worker_group = dist.new_group(worker_ranks, timeout=datetime.timedelta(hours=6))
        num_eval_workers = self.config.get("job.distributed.num_eval_workers")
        eval_worker_ranks = list(range(min_rank, num_eval_workers+min_rank))
        eval_worker_group = dist.new_group(eval_worker_ranks, timeout=datetime.timedelta(hours=6))

        # create parameter server
        server = None
        if self.config.get("job.distributed.parameter_server") == "lapse":
            os.environ["DMLC_NUM_WORKER"] = "0"
            os.environ["DMLC_NUM_SERVER"] = str(self.num_total_workers)
            os.environ["DMLC_ROLE"] = "server"
            os.environ["DMLC_PS_ROOT_URI"] = self.config.get(
                "job.distributed.master_ip"
            )
            os.environ["DMLC_PS_ROOT_PORT"] = str(self.config.get(
                "job.distributed.lapse_port"
            ))

            num_workers_per_server = 1
            lapse.setup(self.num_keys, num_workers_per_server)
            server = lapse.Server(self.num_keys, self.embedding_dim + self.optimizer_dim)
        elif self.config.get("job.distributed.parameter_server") == "shared":
            server = self.parameters

        # create train-worker config, dataset and folder
        device_pool: list = self.config.get("job.device_pool")
        if len(device_pool) == 0:
            device_pool.append(self.config.get("job.device"))
        worker_id = self.rank
        config = deepcopy(self.config)
        config.set("job.device", device_pool[worker_id % len(device_pool)])
        config.folder = os.path.join(self.config.folder, f"worker-{self.rank}")
        config.init_folder()

        parameter_client = KgeParameterClient.create(
            client_type=self.config.get("job.distributed.parameter_server"),
            server_id=0,
            client_id=worker_id + min_rank,
            embedding_dim=self.embedding_dim + self.optimizer_dim,
            server=server,
            num_keys=self.num_keys,
            num_meta_keys=self.num_meta_keys,
            worker_group=worker_group,
            eval_worker_group=eval_worker_group
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
            checkpoint = load_checkpoint(self.checkpoint_name)
            job._load(checkpoint)
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
