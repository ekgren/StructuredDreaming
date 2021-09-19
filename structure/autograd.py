import torch


class BoostGradFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, boost_val):
        ctx.set_materialize_grads(False)
        ctx.boost_val = boost_val
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None
        return grad_output * ctx.boost_val, None


class ClampGradFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, clamp_val):
        ctx.set_materialize_grads(False)
        ctx.clamp_val = clamp_val
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None
        return grad_output.clamp(-ctx.clamp_val, ctx.clamp_val), None


class BoostGrad(torch.nn.Module):
    """
    This class is a PyTorch module that takes a tensor as input and boosts its gradient.

    Parameters
    ----------
    boost_val : float
        The value by which the gradient of the input tensor is boosted.
    """
    def __init__(self, boost_val: float = 1e+1):
        super().__init__()
        self.boost_grad = BoostGradFunc.apply
        self.boost_val = boost_val

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.boost_grad(input, self.boost_val)


class ClampGrad(torch.nn.Module):
    """
    This class is a PyTorch module that takes a tensor as input and clamps its gradient.

    Parameters
    ----------
    clamp_val : float
        The value to clamp the gradients to.

    Methods
    -------
    forward(input)
        Clamps the gradients of the input tensor to be between -clamp_val and clamp_val.
    """
    def __init__(self, clamp_val: float = 1e-6):
        super().__init__()
        self.clamp_grad = ClampGradFunc.apply
        self.clamp_val = clamp_val

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.clamp_grad(input, self.clamp_val)