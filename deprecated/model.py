import torch

from deprecated.autograd import BoostGrad


class ImgBase(torch.nn.Module):
    def __init__(self, size: int = 224, k: float = 5.0, weight_init: float = 0.05):
        super().__init__()
        self.size = size
        self.k = k

        self.color = torch.nn.Parameter(
            torch.tensor(
                [
                    [-0.1409, 0.0855, -0.7620],
                    [0.2596, -0.5239, 0.0996],
                    [0.1653, -0.0719, 0.0889],
                ]
            )
        )
        self.w = torch.nn.Parameter(
            torch.randn(1, 3, size, size, requires_grad=True) * weight_init
        )

    def forward(self) -> torch.Tensor:
        img = self.w
        color = self.color / self.color.norm(p=2)
        img = torch.nn.functional.linear(img.permute(0, 2, 3, 1), color).permute(
            0, 3, 1, 2
        )
        img = self.to_rgb(img, self.k)
        return img

    def to_rgb(self, input: torch.Tensor, k: float) -> torch.Tensor:
        return (input.clamp(-k, k) + k) / (2 * k)


class ImgBaseOld(torch.nn.Module):
    """X"""

    def __init__(self, size=224, weight_init=0.05, decolorize=0.0, darken=0.0):
        super().__init__()
        self.decolorize = decolorize
        self.darken = darken
        self.w = torch.ones(1, 3, size, size, requires_grad=True) * weight_init
        self.w = torch.nn.Parameter(self.w.half())

    def forward(self):
        return self.w

    def post_process(self):
        with torch.no_grad():
            self.w.clamp_(0.0, 1.0)
            if self.decolorize > 0.0:
                self.w += self.decolorize * (
                    -self.w + self.w.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
                )
            if self.darken > 0.0:
                self.w *= 1.0 - self.darken


class ImgBaseFFT(torch.nn.Module):
    """X"""

    def __init__(self, size=224, k=15.0, weight_init=0.05):
        super().__init__()
        self.size = size
        self.k = k
        self.color = torch.nn.Linear(3, 3, bias=False)
        w = torch.fft.rfft2(
            torch.randn(1, 3, size, size, requires_grad=True) * weight_init
        )
        self.w = torch.nn.Parameter(w)
        self.act = torch.sin
        self.bg = BoostGrad()
        self.norm = ChanNorm(dim=3)

    def forward(self):
        img = torch.fft.irfft2(self.w)
        img = self.bg.apply(img)
        img = self.norm(img)
        img = self.color(img.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return img

    def get_img(self, size=None):
        size = size if size is not None else self.size
        img = torch.fft.irfft2(self.w)
        if size != self.size:
            img = torch.nn.functional.interpolate(img, (size, size), mode="area")
        img = self.norm(img)
        img = self.color(img.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return (img.clamp(-self.k, self.k) + self.k) / (2 * self.k)


# From: https://github.com/lucidrains/stylegan2-pytorch
class ChanNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = torch.nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = torch.nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b
