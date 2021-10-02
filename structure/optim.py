import torch
from torch.optim import Optimizer


class SimpleSGD(Optimizer):
    def __init__(self, params, lr=0.1):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                lr = group["lr"]
                grad = p.grad.data
                p.data.add_(grad, alpha=-lr)
        return loss


class ClampSGD(Optimizer):
    def __init__(self, params, lr=0.1, clamp=1e-9):
        defaults = dict(lr=lr, clamp=clamp)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                lr = group["lr"]
                clamp = group["clamp"]
                grad = p.grad.data
                grad.clamp_(-clamp, clamp).div_(clamp)
                p.data.add_(grad, alpha=-lr)
        return loss