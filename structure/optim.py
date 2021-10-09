import torch
from torch.optim import Optimizer


class SimpleSGD(Optimizer):
    def __init__(self, params, lr=0.1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
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


class ClampSGD(torch.optim.Optimizer):
    r""" Clamp SGD
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-1)
        clamp (float, optional): clamping of gradient (default: 1e-30)
        drop (float, optional): dropout of gradient (default: 0)
    """
    def __init__(self, params, lr=1e-1, clamp=1e-30, drop=0.):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= drop < 1.0:
            raise ValueError("Invalid dropout value: {}".format(drop))
        defaults = dict(lr=lr, clamp=clamp, drop=drop)
        super().__init__(params, defaults)

    @torch.no_grad()
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
                drop = group["drop"]
                grad = torch.nn.functional.dropout(p.grad.data.clamp_(-clamp, clamp).div_(clamp), p=drop)
                p.data.add_(grad, alpha=-lr)
        return loss