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
