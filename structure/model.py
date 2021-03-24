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
            if self.decolorize > 0:
                self.w += self.decolorize * \
                          (-self.w + self.w.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1))
            if self.darken > 0:
                self.w *= 1. - self.darken


# TODO: Add progressive growing
# TODO: Add sin model
# TODO: Add Dall-e decoder
# TODO: Add stylegan

