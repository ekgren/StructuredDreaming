import torch
import torchvision

from . import transform


@torch.jit.script
def random_generate_grid(
    l: int = 2, mode: str = "all", device: str = "cuda"
) -> torch.Tensor:
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

    if mode == "all":
        x, _ = torch.rand(l, l, device=device).sort()
        y, _ = torch.rand(l, l, device=device).sort()
        x = (x - x.min(dim=1, keepdim=True).values) / (
            x.max(dim=1, keepdim=True).values - x.min(dim=1, keepdim=True).values
        )
        y = (y - y.min(dim=1, keepdim=True).values) / (
            y.max(dim=1, keepdim=True).values - y.min(dim=1, keepdim=True).values
        )
        x = x.view(-1)
        y = y.t().reshape(-1)
        xy = torch.stack([x, y], dim=1).view(1, l, l, 2) * 2.0 - 1.0
    elif mode == "even":
        coords = torch.arange(l, device=device)
        coords = (coords.float() / (l - 1.0) - 0.5) * 2.0
        offset = torch.rand(1, device=device) * 0.6
        x_offset = offset * (1.0 + 0.05 * (torch.rand(1, device=device) * 2.0 - 1.0))
        y_offset = offset * (1.0 + 0.05 * (torch.rand(1, device=device) * 2.0 - 1.0))
        x = (
            coords * (1.0 - x_offset)
            + (torch.rand(1, device=device) * 2.0 - 1.0) * x_offset
        )
        y = (
            coords * (1.0 - y_offset)
            + (torch.rand(1, device=device) * 2.0 - 1.0) * y_offset
        )
        grid_y, grid_x = torch.meshgrid(y, x)
        xy = torch.stack([grid_x, grid_y], dim=2).view(1, l, l, 2)
    else:  # mode == 'repeat'
        x, _ = torch.rand(l, device=device).sort()
        y, _ = torch.rand(l, device=device).sort()
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())
        grid_y, grid_x = torch.meshgrid(y, x)
        xy = torch.stack([grid_x, grid_y], dim=2).view(1, l, l, 2) * 2.0 - 1.0
    return xy


class ImgSampleBase(torch.nn.Module):
    def __init__(
        self,
        kernel_min: int = 1,
        kernel_max: int = 8,
        grid_size_min: int = 224,
        grid_size_max: int = 448,
        noise: float = 1.0,
        noise_std: float = 0.3,
        cutout: float = 1.0,
        cutout_size: float = 0.25,
        distortion_scale: float = 0.5,
    ):
        super().__init__()
        self.kernel_min = kernel_min
        self.kernel_max = kernel_max
        self.grid_size_min = grid_size_min
        self.grid_size_max = grid_size_max
        self.noise = noise
        self.noise_std = noise_std
        self.cutout = cutout
        self.cutout_size = cutout_size
        self.perspective_transformer = T.RandomPerspective(
            distortion_scale=distortion_scale,
            p=1.0,
            interpolation=T.InterpolationMode.BILINEAR,
        )

    def forward(
        self, input: torch.Tensor, size: int = 224, bs: int = 1
    ) -> torch.Tensor:
        imgs = []

        for _ in range(bs):
            # Draw kernel size from uniform x uniform distribution
            kernel_size = int(
                torch.randint(
                    self.kernel_min,
                    int(torch.randint(self.kernel_min + 1, self.kernel_max, ())),
                    (),
                ).item()
            )

            # Generate grid for sampling
            grid_mode_idx = int(torch.randint(1, 3, ()).item())
            grid_mode = ("all", "even", "repeat")
            grid_size = int(
                self.grid_size_min
                + torch.rand(()).item() * (self.grid_size_max - self.grid_size_min)
            )
            grid = random_generate_grid(grid_size, mode=grid_mode[grid_mode_idx])

            # Sample original input
            img = torch.nn.functional.grid_sample(
                input, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )
            img = torch.nn.functional.interpolate(img, (size, size), mode="area")
            imgs.append(img)

            # Transform, downsize and sample original input
            img = transform.noise(input, noise=self.noise, noise_std=self.noise_std)
            img = transform.cutout(
                img, cutout=self.cutout, cutout_size=self.cutout_size
            )
            img = torch.nn.functional.avg_pool2d(
                img, kernel_size=kernel_size, stride=kernel_size, padding=0
            )
            img = torch.nn.functional.grid_sample(
                img, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )
            img = torch.nn.functional.interpolate(img, (size, size), mode="area")
            imgs.append(img)

        return torch.cat(imgs, dim=0)


class ImgSampleStylegan(torch.nn.Module):
    def __init__(
        self,
        kernel_min: int = 1,
        kernel_max: int = 8,
        grid_size_min: int = 224,
        grid_size_max: int = 448,
        noise: float = 1.0,
        noise_std: float = 0.3,
        cutout: float = 1.0,
        cutout_size: float = 0.25,
    ):
        super().__init__()
        self.kernel_min = kernel_min
        self.kernel_max = kernel_max
        self.grid_size_min = grid_size_min
        self.grid_size_max = grid_size_max
        self.noise = noise
        self.noise_std = noise_std
        self.cutout = cutout
        self.cutout_size = cutout_size

    def forward(
        self, input: torch.Tensor, size: int = 224, bs: int = 1
    ) -> torch.Tensor:
        imgs = []

        for _ in range(bs):
            # Draw kernel size from uniform x uniform distribution
            kernel_size = int(
                torch.randint(
                    self.kernel_min,
                    int(torch.randint(self.kernel_min + 1, self.kernel_max, ())),
                    (),
                ).item()
            )

            # Generate grid for sampling
            grid_mode_idx = int(torch.randint(1, 3, ()).item())
            grid_mode = ("all", "even", "repeat")
            grid_size = int(
                self.grid_size_min
                + torch.rand(()).item() * (self.grid_size_max - self.grid_size_min)
            )
            grid = random_generate_grid(grid_size, mode=grid_mode[grid_mode_idx])

            # Sample original input
            img = torch.nn.functional.grid_sample(
                input, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )
            img = torch.nn.functional.interpolate(img, (size, size), mode="area")
            imgs.append(img)

            # Transform, downsize and sample original input
            img = transform.noise(input, noise=self.noise, noise_std=self.noise_std)
            img = transform.cutout(
                img, cutout=self.cutout, cutout_size=self.cutout_size
            )
            img = torch.nn.functional.avg_pool2d(
                img, kernel_size=kernel_size, stride=kernel_size, padding=0
            )
            img = torch.nn.functional.grid_sample(
                img, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )
            img = torch.nn.functional.interpolate(img, (size, size), mode="area")
            imgs.append(img)

        return torch.cat(imgs, dim=0)
