import torch


class ImgBase(torch.nn.Module):
    """ X """
    def __init__(self, size=224, weight_init=0.05, decolorize=0., darken=0.):
        super().__init__()
        self.decolorize = decolorize
        self.darken = darken
        self.w = torch.ones(1, 3, size, size, requires_grad=True) * weight_init
        self.w = torch.nn.Parameter(self.w.half())

    def forward(self):
        return self.w

    def post_process(self):
        with torch.no_grad():
            self.w.clamp_(0., 1.)
            if self.decolorize > 0.:
                self.w += self.decolorize * \
                          (-self.w + self.w.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1))
            if self.darken > 0.:
                self.w *= 1. - self.darken


class ImgBaseFFT(torch.nn.Module):
    """ X """
    def __init__(self, size=224, k=15., weight_init=0.05):
        super().__init__()
        self.size = size
        self.k = k
        w = torch.fft.rfft2(torch.randn(1, 3, size, size, requires_grad=True) * weight_init)
        self.w = torch.nn.Parameter(w)
        self.act = torch.sin

    def forward(self):
        return torch.fft.irfft2(self.w)

    def get_img(self, size=None):
        size = size if size is not None else self.size
        img_out = torch.fft.irfft2(self.w)
        if size != self.size:
            img_out = torch.nn.functional.interpolate(img_out, (size, size), mode='area')
        return (self.act(img_out / self.k) + 1.) / 2.

