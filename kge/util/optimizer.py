from kge import Config, Configurable
import torch.optim
from torch.optim.lr_scheduler import _LRScheduler
import re
from operator import or_
from functools import reduce
from kge.util.dist_sgd import DistSGD
from kge.util.dist_adagrad import DistAdagrad


class KgeOptimizer:
    """ Wraps torch optimizers """

    @staticmethod
    def create(config, model, parameter_client=None, lapse_indexes=None):
        """ Factory method for optimizer creation """
        if config.get("train.optimizer.default.type") == "dist_sgd":
            optimizer = DistSGD(
                model,
                parameter_client=parameter_client,
                lapse_indexes=lapse_indexes,
                **config.get("train.optimizer.default.args"),
            )
            return optimizer
        if config.get("train.optimizer.default.type") in ["dist_adagrad", "dist_rowadagrad"]:
            from kge.distributed.misc import get_min_rank
            is_row = False
            use_lr_scheduler = False
            if config.get("train.optimizer.default.type") == "dist_rowadagrad":
                is_row = True
            if config.get("train.lr_scheduler") != "":
                use_lr_scheduler = True
            min_rank = get_min_rank(config)
            optimizer = DistAdagrad(
                KgeOptimizer._get_parameters_and_optimizer_args(
                    config,
                    model,
                    distributed=True
                ),
                parameter_client=parameter_client,
                lapse_indexes=lapse_indexes,
                lapse_optimizer_index_offset=model.dataset.num_entities() + model.dataset.num_relations(),
                async_write_back=[config.get("job.distributed.entity_async_write_back"),
                                  config.get("job.distributed.relation_async_write_back")],
                is_row=is_row,
                use_lr_scheduler=use_lr_scheduler,
                min_rank=min_rank,
                **config.get("train.optimizer.default.args"),
            )
            return optimizer
        else:
            try:
                optimizer = getattr(torch.optim, config.get("train.optimizer.default.type"))
                return optimizer(
                    KgeOptimizer._get_parameters_and_optimizer_args(config, model),
                    **config.get("train.optimizer.default.args"),
                )
            except AttributeError:
                # perhaps TODO: try class with specified name -> extensibility
                raise ValueError(
                    f"Could not create optimizer {config.get('train.optimizer')}. "
                    f"Please specify an optimizer provided in torch.optim"
                )

    @staticmethod
    def _get_parameters_and_optimizer_args(config, model, distributed=False):
        """
        Group named parameters by regex strings provided with optimizer args.
        Constructs a list of dictionaries of the form:
        [
            {
                "name": name of parameter group
                "params": list of parameters to optimize
                # parameter specific options as for example learning rate
                ...
            },
            ...
        ]
        """

        named_parameters = dict(model.named_parameters())
        optimizer_settings = config.get("train.optimizer")
        parameter_names_per_search = dict()
        # filter named parameters by regex string
        for group_name, parameter_group in optimizer_settings.items():
            if group_name == "default":
                continue
            if "type" in parameter_group.keys():
                raise NotImplementedError("Multiple optimizer types are not yet supported.")
            search_pattern = re.compile(parameter_group["regex"])
            filtered_named_parameters = set(
                filter(search_pattern.match, named_parameters.keys())
            )
            parameter_names_per_search[group_name] = filtered_named_parameters

        # check if something was matched by multiple strings
        parameter_values = list(parameter_names_per_search.values())
        for i, (group_name, param) in enumerate(parameter_names_per_search.items()):
            for j in range(i + 1, len(parameter_names_per_search)):
                intersection = set.intersection(param, parameter_values[j])
                if len(intersection) > 0:
                    raise ValueError(
                        f"The parameters {intersection}, were matched by the optimizer "
                        f"group {group_name} and {list(parameter_names_per_search.keys())[j]}"
                    )
        resulting_parameters = []
        for group_name, params in parameter_names_per_search.items():
            optimizer_settings[group_name]["args"]["params"] = [
                named_parameters[param] for param in params
            ]
            optimizer_settings[group_name]["args"]["name"] = group_name
            if distributed and (group_name == "entity" or group_name == "relation"):
                optimizer_settings[group_name]["args"][
                    "async_write_back"] = config.get(
                    f"job.distributed.{group_name}_async_write_back")
                optimizer_settings[group_name]["args"]["sync_level"] = config.get(
                    f"job.distributed.{group_name}_sync_level")
                optimizer_settings[group_name]["args"][
                    "local_to_lapse_mapper"] = getattr(
                    model,
                    f"_{group_name}_embedder").local_to_lapse_mapper
                optimizer_settings[group_name]["args"]["optimizer_values"] = getattr(
                    model,
                    f"_{group_name}_embedder"
                ).optimizer_values
            resulting_parameters.append(optimizer_settings[group_name]["args"])

        # add unmatched parameters to default group
        if len(parameter_names_per_search) > 0:
            default_parameter_names = set.difference(
                set(named_parameters.keys()),
                reduce(or_, list(parameter_names_per_search.values())),
            )
            default_parameters = [
                named_parameters[default_parameter_name]
                for default_parameter_name in default_parameter_names
            ]
            resulting_parameters.append(
                {"params": default_parameters, "name": "default"}
            )
        else:
            # no parameters matched, add everything to default group
            resulting_parameters.append(
                {"params": model.parameters(), "name": "default"}
            )
        return resulting_parameters


class KgeLRScheduler(Configurable):
    """ Wraps torch learning rate (LR) schedulers """

    def __init__(self, config: Config, optimizer):
        super().__init__(config)
        name = config.get("train.lr_scheduler")
        args = config.get("train.lr_scheduler_args")
        self._lr_scheduler: _LRScheduler = None
        if name != "":
            # check for consistency of metric-based scheduler
            self._metric_based = name in ["ReduceLROnPlateau"]
            if self._metric_based:
                desired_mode = "max" if config.get("valid.metric_max") else "min"
                if "mode" in args:
                    if args["mode"] != desired_mode:
                        raise ValueError(
                            (
                                "valid.metric_max ({}) and train.lr_scheduler_args.mode "
                                "({}) are inconsistent."
                            ).format(config.get("valid.metric_max"), args["mode"])
                        )
                    # all fine
                else:  # mode not set, so set it
                    args["mode"] = desired_mode
                    config.set("train.lr_scheduler_args.mode", desired_mode, log=True)

            # create the scheduler
            try:
                self._lr_scheduler = getattr(torch.optim.lr_scheduler, name)(
                    optimizer, **args
                )
            except Exception as e:
                raise ValueError(
                    (
                        "Invalid LR scheduler {} or scheduler arguments {}. "
                        "Error: {}"
                    ).format(name, args, e)
                )

    def step(self, metric=None):
        if self._lr_scheduler is None:
            return
        if self._metric_based:
            if metric is not None:
                # metric is set only after validation has been performed, so here we
                # step
                self._lr_scheduler.step(metrics=metric)
        else:
            # otherwise, step after every epoch
            self._lr_scheduler.step()

    def state_dict(self):
        if self._lr_scheduler is None:
            return dict()
        else:
            return self._lr_scheduler.state_dict()

    def load_state_dict(self, state_dict):
        if self._lr_scheduler is None:
            pass
        else:
            self._lr_scheduler.load_state_dict(state_dict)
