import torch


@torch.jit.script
def random_generate_grid(l: int = 2,
                         mode: str = 'all',
                         device: str = 'cuda') -> torch.Tensor:
    """
    Generates a grid of coordinates in the range [-1, 1] to be used with pytorch
    grid_sample.

    Parameters
    ----------
    l : int
        The number of coordinates along each dimension.
    mode : str
        'all', 'even' or 'repeat'. If 'all', generates a grid of size l*l.
        If 'even', generates a grid of size l*l.
        If 'repeat', generates a grid of size 1*l.
    device : str
        'cuda' or 'cpu'.

    Returns
    -------
    xy : torch.Tensor
        A tensor of shape (l*l, 2) if mode == 'all', or (l*l, 2) if mode == 'repeat'.
    """
    if mode != "all" and mode != "even" and mode != "repeat":
        raise ValueError(
            "random_generate_grid(): expected mode to be "
            "'all' or 'repeat', but got: '{}'".format(mode)
        )

    if mode == 'all':
        x, _ = torch.rand(l, l, device=device).sort()
        y, _ = torch.rand(l, l, device=device).sort()
        x = (x - x.min(dim=1, keepdim=True).values)/(x.max(dim=1, keepdim=True).values - x.min(dim=1, keepdim=True).values)
        y = (y - y.min(dim=1, keepdim=True).values)/(y.max(dim=1, keepdim=True).values - y.min(dim=1, keepdim=True).values)
        x = x.view(-1)
        y = y.t().reshape(-1)
        xy = torch.stack([x, y], dim=1).view(1, l, l, 2) * 2. - 1.
    elif mode == 'even':
        coords = torch.arange(l, device=device)
        coords = (coords.float()/(l-1.)-0.5)*2.
        offset = torch.rand(1, device=device)*0.6
        x_offset = offset * (1. + 0.05*(torch.rand(1, device=device)*2.-1.))
        y_offset = offset * (1. + 0.05*(torch.rand(1, device=device)*2.-1.))
        x = coords * (1. - x_offset) + (torch.rand(1, device=device)*2.-1.) * x_offset
        y = coords * (1. - y_offset) + (torch.rand(1, device=device)*2.-1.) * y_offset
        grid_y, grid_x = torch.meshgrid(y, x)
        xy = torch.stack([grid_x, grid_y], dim=2).view(1, l, l, 2)
    else: # mode == 'repeat'
        x, _ = torch.rand(l, device=device).sort()
        y, _ = torch.rand(l, device=device).sort()
        x = (x - x.min())/(x.max()-x.min())
        y = (y - y.min())/(y.max()-y.min())
        grid_y, grid_x = torch.meshgrid(y, x)
        xy = torch.stack([grid_x, grid_y], dim=2).view(1, l, l, 2) * 2. - 1.
    return xy