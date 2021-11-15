from collections import deque
import torch
import numpy as np
from torch.optim.optimizer import Optimizer
from copy import deepcopy


class DistAdagrad(Optimizer):
    """Implements Adagrad algorithm.

    It has been proposed in `Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)

    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html
    """

    def __init__(
        self,
        # model,
        params,
        lr=1e-2,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-10,
        parameter_client=None,
        lapse_indexes=None,
        lapse_optimizer_index_offset=0,
        async_write_back=[],
        is_row=False,
        use_lr_scheduler=False,
        min_rank=-1,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(
                "Invalid initial_accumulator_value value: {}".format(
                    initial_accumulator_value
                )
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        self.lapse_optimizer_index_offset = lapse_optimizer_index_offset
        self.lapse_indexes = lapse_indexes
        self.pulled_parameters = [None, None]
        self.async_write_back = async_write_back

        self.is_row = is_row

        self.parameter_client = parameter_client
        # this array stores helper cpu tensors in which we pull data from the parameter
        # client. We don't want to create a new tensor in every step.
        self.pull_tensors = {"entity": None, "relation": None}
        self.push_keys = {"entity": None, "relation": None}
        self.push_tensors = {
            "entity": None,
            "relation": None,
        }
        self.use_lr_scheduler = use_lr_scheduler
        self.min_rank = min_rank
        self.entity_async_wait_values = deque()
        self.relation_async_wait_values = deque()
        self.async_wait_values = {
            "entity": self.entity_async_wait_values,
            "relation": self.relation_async_wait_values,
        }

        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            eps=eps,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
        )
        super(DistAdagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            if group["name"] != "default":
                if parameter_client.get_lr(group["name"]) == 0:
                    self.parameter_client.set_lr(group["name"], group["lr"])
            group["prev_lr"] = group["lr"]
            for i, p in enumerate(group["params"]):
                state = self.state[p]
                state["step"] = 0
                # state["sum"] = self.optimizer_values[i]

    def share_memory(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["sum"].share_memory_()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # we need to wait here for the previous push to finish, otherwise we can not
        #  delete the push tensors
        if self.async_write_back[0]:
            for wait_value in self.entity_async_wait_values:
                self.parameter_client.wait(wait_value)
            self.entity_async_wait_values.clear()
        if self.async_write_back[1]:
            for wait_value in self.relation_async_wait_values:
                self.parameter_client.wait(wait_value)
            self.relation_async_wait_values.clear()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if group["lr_decay"] > 0:
                    # we only need to synchronize steps between workers if we actually
                    #  use the step variable for something
                    self.parameter_client.step_optim(i)
                    state["step"] = self.parameter_client.get_step_optim(group["name"])
                else:
                    state["step"] += 1
                if self.use_lr_scheduler:
                    if self.parameter_client.rank == self.min_rank:
                        if group["prev_lr"] != group["lr"]:
                            self.parameter_client.set_lr(group["name"], group["lr"])
                    group["lr"] = self.parameter_client.get_lr(group["name"])

                if group["weight_decay"] != 0:
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            "weight_decay option is not compatible with sparse gradients"
                        )
                    grad = grad.add(p, alpha=group["weight_decay"])

                clr = group["lr"] / (1 + (state["step"] - 1) * group["lr_decay"])

                if grad.is_sparse:
                    grad = (
                        grad.coalesce()
                    )  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()[0]
                    # grad_indices_flat = grad_indices.flatten()
                    grad_values = grad._values()
                    size = grad.size()

                    # pull the current internal optimizer parameters
                    state_sum = group["optimizer_values"][grad_indices]

                    if not self.is_row:
                        sum_update_values = grad_values.pow(2)
                    else:
                        sum_update_values = grad_values.pow(2).mean(1).view(-1, 1)
                    state_sum.add_(sum_update_values)
                    # state["sum"].add_(make_sparse(sum_update_values))
                    if group["sync_level"] == "batch":
                        pass
                    else:
                        # state["sum"][grad_indices_flat] = state_sum
                        group["optimizer_values"][grad_indices] = state_sum

                    # std = state["sum"].sparse_mask(grad)
                    # std_values = std._values().sqrt_().add_(group["eps"])
                    std_values = state_sum.sqrt_().add_(group["eps"])
                    update_value = (grad_values / std_values).mul_(-clr)
                    if group["sync_level"] == "batch":
                        update_indexes = grad_indices.cpu()
                        self.push_keys[group["name"]] = group["local_to_lapse_mapper"][
                            update_indexes
                        ]
                        unnecessary_dim = (
                            self.parameter_client.dim
                            - update_value.shape[1]
                            - sum_update_values.shape[1]
                        )
                        if unnecessary_dim > 0:
                            self.push_tensors[group["name"]] = torch.cat(
                                (
                                    update_value,
                                    sum_update_values,
                                    torch.empty(
                                        (len(update_value), unnecessary_dim),
                                        device=update_value.device,
                                    ),
                                ),
                                dim=1,
                            ).cpu()
                        else:
                            self.push_tensors[group["name"]] = torch.cat(
                                (update_value, sum_update_values), dim=1
                            ).cpu()
                        self.async_wait_values[group["name"]].append(
                            self.parameter_client.push(
                                self.push_keys[group["name"]],
                                self.push_tensors[group["name"]],
                                asynchronous=True,
                                # asynchronous=self.async_write_back[i]
                            )
                        )
                    else:
                        p.data.index_add_(0, grad_indices, update_value)
                        # p.add_(make_sparse(update_value))
                    # p.add_(make_sparse(grad_values / std_values), alpha=-clr)
                else:
                    raise ValueError(
                        "Currently only sparse parameters supported with dist_adagrad and dist_rowadagrad"
                    )

        return loss

    def pull_all(self):
        """
        loads optimizer values stored in distributed lookup embedder to state[sum]
        used for checkpoint of complete model
        embedder.pull_all needs to be called before this function.
        """
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                self.state[p]["sum"] = group["optimizer_values"]
                self.state[p]["step"] = self.parameter_client.get_step_optim(
                    group["name"]
                )
            if group["name"] == "default":
                continue
            group["lr"] = self.parameter_client.get_lr(group["name"])

    def state_dict(self) -> dict:
        """
        We are removing the optimizer values from state dict since stored separately
        """
        state_dict = super(DistAdagrad, self).state_dict()
        for i, group in enumerate(state_dict["param_groups"]):
            for key in ["optimizer_values", "local_to_lapse_mapper"]:
                state_dict["param_groups"][i].pop(key, None)
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """
        We need to keep the created references to the opitmizer values in the embedder.
        super.load_state_dict removes the created references if not in state_dict.
        """

        saved_references = list()
        for group in self.param_groups:
            ref = dict()
            if "optimizer_values" in group:
                ref["optimizer_values"] = group["optimizer_values"]
            if "local_to_lapse_mapper" in group:
                ref["local_to_lapse_mapper"] = group["local_to_lapse_mapper"]
            saved_references.append(ref)
        super(DistAdagrad, self).load_state_dict(state_dict)
        for ref, group in zip(saved_references, self.param_groups):
            group.update(ref)
            if group["name"] != "default":
                if self.parameter_client.get_lr(group["name"]) == 0:
                    self.parameter_client.set_lr(group["name"], group["lr"])
