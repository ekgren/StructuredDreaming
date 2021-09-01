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


class ClampGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        return grad_output.clamp(-0.0005, 0.0005)