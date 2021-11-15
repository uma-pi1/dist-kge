import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required


class DistSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    """

    def __init__(
        self,
        model,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        parameter_client=None,
        lapse_indexes=None,
        local_index_mappers=None,
    ):
        params = [p for p in model.parameters() if p.requires_grad]
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        self.lapse_indexes = lapse_indexes
        # self.local_index_mappers = local_index_mappers
        self.local_index_mappers = [
            model._entity_embedder.local_index_mapper,
            model._relation_embedder.local_index_mapper,
        ]
        self.local_to_lapse_mappers = [
            model._entity_embedder.local_to_lapse_mapper,
            model._relation_embedder.local_to_lapse_mapper,
        ]
        self.parameter_client = parameter_client
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(DistSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DistSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                if d_p.is_sparse:
                    d_p = (
                        d_p.coalesce()
                    )  # the update is non-linear so indices must be unique
                    push_tensor = d_p._values().mul_(-group["lr"]).cpu()
                    update_indexes = d_p._indices().cpu()
                    push_keys = self.local_to_lapse_mappers[i][update_indexes].view(-1)
                    self.parameter_client.push(push_keys, push_tensor)
                else:
                    indexes_to_push_mask = self.local_to_lapse_mappers[i] != -1
                    self.parameter_client.push(
                        self.local_to_lapse_mappers[i][indexes_to_push_mask],
                        (-group["lr"] * d_p).cpu()[indexes_to_push_mask],
                    )
                # self.lapse_worker.push(self.lapse_indexes[i], (-group['lr']*d_p).cpu().to_dense().numpy())
                # p.add_(d_p, alpha=-group['lr'])

        return loss

    def pull_entities(self, entity_ids):
        pass

    def pull_relations(self, relation_ids):
        pass

    def set_entities(self):
        pass

    def set_relations(self):
        pass

    def pull_all(self):
        pass

    def push_all(self):
        pass
