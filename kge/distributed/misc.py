import datetime

from torch import distributed as dist

from kge import Config


def get_min_rank(config: Config):
    if config.get("job.distributed.parameter_server") in ["shared", "lapse"]:
        # with a shared parameter server we don't create an additional process
        min_rank = 1
    else:
        min_rank = 2
    # with parallel evaluation we do need the scheduler again
    # if config.get("job.type") in ["valid", "test", "eval"]:
    #     # we do not need a scheduler, therefore reduce min_rank
    #     min_rank -= 1
    return min_rank


def get_optimizer_dim(config: Config, dim):
    optimizer = config.get("train.optimizer.default.type")
    if optimizer == "dist_sgd":
        optimizer_dim = -1
    elif optimizer == "dist_adagrad":
        optimizer_dim = dim
    elif optimizer == "dist_rowadagrad":
        optimizer_dim = 1
    else:
        raise NotImplementedError(f"Optimizer {optimizer} not implemented in distributed setting")
    return optimizer_dim


def get_num_meta_keys(config):
    num_meta_keys = 3
    if config.get("train.optimizer.default.type") in [
        "dist_adagrad",
        "dist_rowadagrad",
    ]:
        num_meta_keys += 2
    return num_meta_keys


def get_num_keys(config, dataset):
    num_keys = dataset.num_entities() + dataset.num_relations()
    num_keys += get_num_meta_keys(config)
    return num_keys


# def get_num_keys(config, dataset):
#     num_keys = dataset.num_entities() + dataset.num_relations()
#     num_meta_keys = 3
#     if config.get("train.optimizer.default.type") in [
#         "dist_adagrad",
#         "dist_rowadagrad",
#     ]:
#         #    num_keys *= 2
#         num_meta_keys += 2
#     num_keys += num_meta_keys
#
#     return get_num_keys(config, dataset), get_num_meta_keys(config)


def initialize_worker_groups(config, rank):
    min_rank = get_min_rank(config)
    world_size = config.get("job.distributed.num_workers") + min_rank
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(hours=6),
    )
    worker_ranks = list(range(min_rank, world_size))
    worker_group = dist.new_group(worker_ranks, timeout=datetime.timedelta(hours=6))
    num_eval_workers = config.get("job.distributed.num_eval_workers")
    eval_worker_ranks = list(range(min_rank, num_eval_workers + min_rank))
    eval_worker_group = dist.new_group(eval_worker_ranks, timeout=datetime.timedelta(hours=6))
    return worker_group, eval_worker_group
