import os
import time
import logging
import warnings

import psutil
from signal import signal, SIGINT
from py3nvml.py3nvml import *
from typing import Dict, Optional
from kge import Config, Dataset
from kge.distributed.parameter_server import init_torch_server, init_lapse_scheduler
from kge.distributed.worker_process import WorkerProcessPool
from kge.distributed.work_scheduler import WorkScheduler
from kge.distributed.misc import get_optimizer_dim, get_min_rank

import torch
from torch import multiprocessing as mp


def monitor_hardware(folder, interval=1):
    def bytes_to_mb(bytes_amount):
        return round(bytes_amount / 1024 / 1024, 2)

    logger = logging.getLogger("hardware_monitor")
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(folder, "hardware_monitor.log"))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # let's monitor the default connection between OUR two servers
    # todo: just monitor all interfaces later on
    interface = "enp130s0f0"
    while True:
        time.sleep(interval)
        cpu_percentage = psutil.cpu_percent()
        memory_percentage = psutil.virtual_memory().percent
        network_info = psutil.net_io_counters()
        bytes_sent = network_info.bytes_sent
        bytes_recv = network_info.bytes_recv
        # timestamp;cpu%;mem%;net_sent;net_recvm

        msg = f"{time.time()};{cpu_percentage};{memory_percentage};{bytes_to_mb(bytes_sent)};{bytes_to_mb(bytes_recv)}"
        network_info = psutil.net_io_counters(pernic=True)
        if interface in network_info.keys():
            bytes_sent = network_info[interface].bytes_sent
            bytes_recv = network_info[interface].bytes_recv
            msg += f";{bytes_to_mb(bytes_sent)};{bytes_to_mb(bytes_recv)}"
        logger.info(
            msg=msg
        )


def monitor_gpus(folder, interval=1):
    try:
        nvmlInit()
    except Exception:
        print("could not initialize GPU monitor")
        return
    device_count = nvmlDeviceGetCount()
    if device_count == 0:
        return
    logger = logging.getLogger("gpu_monitor")
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(folder, "gpu_monitor.log"))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    while True:
        time.sleep(interval)
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            proc_res = nvmlDeviceGetComputeRunningProcesses(handle)
            mem_per_process = list(
                map(lambda obj: (obj.pid, obj.usedGpuMemory), proc_res)
            )
            res = nvmlDeviceGetUtilizationRates(handle)
            mem_res = nvmlDeviceGetMemoryInfo(handle)
            # timestamp;device_id;gpu_util;gpu_mem_util;gpu_temp;mem_per_process
            logger.info(
                f"{time.time()};{i};{res.gpu};{round((mem_res.used/mem_res.total)*100)};{mem_per_process}"
            )


def create_and_run_distributed(
    config: Config, dataset: Optional[Dataset] = None, checkpoint: Optional[Dict] = None
):
    # setting num eval workers to 1 if < 1
    if config.get("job.distributed.num_eval_workers") < 1:
        warnings.warn("Need to have at least one worker for evaluation."
                      "Setting job.distributed.num_eval_workers to 1")
        config.set("job.distributed.num_eval_workers", 1)
    # setting num workers to 1 if < 1
    if config.get("job.distributed.num_workers") < 1:
        warnings.warn("Need to have at least one worker for training."
                      "Setting job.distribtued.num_workers to 1")
        config.set("job.distributed.num_workers", 1)
    # specific settings for valid only jobs
    if config.get("job.type") in ["valid", "test", "eval"]:
        config.set("job.distributed.parameter_server", "shared")
        num_eval_workers = config.get("job.distributed.num_eval_workers")
        config.set("job.distributed.num_workers", num_eval_workers)
        config.set("job.distributed.num_workers_machine", num_eval_workers)
        config.set("job.distributed.num_machines", 1)
        config.set("job.distributed.gloo_socket_ifname", "lo")
        config.set("job.distributed.master_ip", "127.0.0.1")
        config.set(f"{config.get('model')}.create_eval", True)
    os.environ["OMP_NUM_THREADS"] = str(
        config.get("job.distributed.num_threads_per_process")
    )
    os.environ["GLOO_SOCKET_IFNAME"] = config.get("job.distributed.gloo_socket_ifname")
    processes = []
    num_keys = dataset.num_entities() + dataset.num_relations()
    num_meta_keys = 3
    num_workers = config.get("job.distributed.num_workers")
    master_ip = config.get("job.distributed.master_ip")
    master_port = config.get("job.distributed.master_port")
    lapse_port = config.get("job.distributed.lapse_port")
    num_partitions = config.get("job.distributed.num_partitions")
    min_rank = get_min_rank(config)
    dist_world_size = num_workers + min_rank
    dim = config.get("lookup_embedder.dim")
    optimizer_dim = get_optimizer_dim(config, dim)
    if config.get("train.optimizer.default.type") in [
        "dist_adagrad",
        "dist_rowadagrad",
    ]:
        #    num_keys *= 2
        num_meta_keys += 2
    # meta keys. contains for example a variable indicating whether to stop or
    #  not
    num_keys += num_meta_keys

    if (
        config.get("job.distributed.repartition_epoch")
        and config.get("job.distributed.partition_type") == "stratification"
    ):
        # with stratificaton we have a lot of open files that need to be shared
        # between processes. Some servers don't allow that. Therefore set sharing
        # strategy to file_system to avoid too many open files error
        torch.multiprocessing.set_sharing_strategy("file_system")

    # catch interrupt (to shut down lapse and other processes)
    processes = []
    monitoring_processes = []
    worker_process_pool = None

    def kill_processes(signal_received, frame):
        print("\nSIGINT or CTRL-C detected. Shutting down all processes and exiting...")
        for process in processes:
            if process is not None:
                try:
                    process.kill()
                except AttributeError:
                    print("process already killed")
        for process in monitoring_processes:
            if process is not None:
                process.kill()
        if worker_process_pool is not None:
            worker_process_pool.kill()
        exit(0)
    signal(SIGINT, kill_processes)

    if config.get("job.type") == "train":
        # start hardware monitoring
        monitor_process = mp.Process(
            target=monitor_hardware, args=(config.folder, 0.5), daemon=True
        )
        monitoring_processes.append(monitor_process)
        monitor_process.start()
        gpu_monitor_process = mp.Process(
            target=monitor_gpus, args=(config.folder, 1), daemon=True
        )
        monitoring_processes.append(gpu_monitor_process)
        gpu_monitor_process.start()



    if config.get("job.distributed.machine_id") == 0:
        if config.get("job.distributed.parameter_server") == "lapse":
            p = mp.Process(
                target=init_lapse_scheduler,
                args=(
                    num_workers,
                    num_keys,
                    master_ip,
                    master_port,
                    lapse_port,
                    dist_world_size,
                    min_rank,
                ),
                daemon=True,
            )
            processes.append(p)
            p.start()
        elif config.get("job.distributed.parameter_server") == "torch":
            p = mp.Process(
                target=init_torch_server,
                args=(
                    num_workers,
                    num_keys,
                    dim + optimizer_dim,
                    master_ip,
                    master_port,
                    min_rank,
                    config.get("job.distributed.num_eval_workers")
                ),
                daemon=True,
            )
            processes.append(p)
            p.start()

        # create a work scheduler
        if config.get("job.type") != "train":
            partition_type = "random"
        else:
            partition_type = config.get("job.distributed.partition_type")
        print("init scheduler")
        scheduler_init_time = time.time()
        scheduler = WorkScheduler.create(config=config, partition_type=partition_type, dataset=dataset)
        config.log(f"scheduler initialized after: {time.time()-scheduler_init_time}")
        print("start scheduler")
        scheduler_start_time = time.time()
        processes.append(scheduler)
        scheduler.start()
        config.log(f"scheduler start took: {time.time()-scheduler_start_time}")

    # create all train-workers in a worker pool
    num_workers = config.get("job.distributed.num_workers")
    num_workers_machine = config.get("job.distributed.num_workers_machine")
    if num_workers_machine <= 0:
        num_workers_machine = num_workers
    already_init_workers = config.get("job.distributed.already_init_workers")
    if already_init_workers < 0:
        already_init_workers = config.get("job.distributed.machine_id") * config.get("job.distributed.num_workers_machine")
    worker_process_pool = WorkerProcessPool(
        num_workers,
        num_workers_machine,
        already_init_workers,
        num_keys,
        num_meta_keys,
        dim,
        optimizer_dim,
        config,
        dataset,
        checkpoint,
    )
    valid_trace = worker_process_pool.join()
    for p in processes:
        p.join()

    if config.get("job.type") == "train":
        monitor_process.terminate()
        gpu_monitor_process.terminate()
    return valid_trace
