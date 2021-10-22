import torch


@torch.jit.script
def noise(
    input: torch.Tensor,
    noise: float = 0.0,
    noise_std: float = 0.1,
    p: float = 1.0,
    device: str = "cuda",
) -> torch.Tensor:
    """Apply additive RGB noise with probability (noise * p)."""
    if noise > 0:
        batch_size, num_channels, height, width = input.shape
        sigma = torch.randn([batch_size, 1, 1, 1], device=device).abs() * noise_std
        sigma = torch.where(
            torch.rand([batch_size, 1, 1, 1], device=device) < noise * p,
            sigma,
            torch.zeros_like(sigma),
        )
        input = (
            input
            + torch.randn([batch_size, num_channels, height, width], device=device)
            * sigma
        )
    return input


@torch.jit.script
def cutout(
    input: torch.Tensor,
    cutout: float = 0.0,
    cutout_size: float = 0.1,
    p: float = 1.0,
    device: str = "cuda",
) -> torch.Tensor:
    """Apply cutout with probability (cutout * p)."""
    if cutout > 0:
        batch_size, num_channels, height, width = input.shape
        size = torch.full([batch_size, 2, 1, 1, 1], cutout_size, device=device)
        size = torch.where(
            torch.rand([batch_size, 1, 1, 1, 1], device=device) < cutout * p,
            size,
            torch.zeros_like(size),
        )
        center = torch.rand([batch_size, 2, 1, 1, 1], device=device)
        coord_x = torch.arange(width, device=device).reshape([1, 1, 1, -1])
        coord_y = torch.arange(height, device=device).reshape([1, 1, -1, 1])
        mask_x = ((coord_x + 0.5) / width - center[:, 0]).abs() >= size[:, 0] / 2
        mask_y = ((coord_y + 0.5) / height - center[:, 1]).abs() >= size[:, 1] / 2
        mask = torch.logical_or(mask_x, mask_y).to(torch.float32)
        input = input * mask
    return input


@torch.jit.script
def color(input: torch.Tensor,
            c: torch.Tensor,
            c_bias: torch.Tensor,
            c_scale: torch.Tensor) -> torch.Tensor:
    c = torch.sin(c)
    c = c / (c.norm() + 1e-8)
    c_bias = torch.sin(c_bias)
    c_bias = c_bias / (c_bias.norm() + 1e-8)
    input = input.permute(0, 2, 3, 1)
    input = torch.nn.functional.linear(input, c, bias=c_bias)
    input = input.permute(0, 3, 1, 2) * c_scale.abs()
    input = (torch.sin(input) + 1.)/2.
    return input
