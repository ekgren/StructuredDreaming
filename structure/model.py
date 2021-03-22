import torch


class ModelSmall(torch.nn.Module):
    """ X """
    def __init__(self, size=224, weight_init=0.05):
        super().__init__()
        self.w = torch.ones(1, 3, size, size, requires_grad=True) * weight_init
        self.w = torch.nn.Parameter(self.w)

    def forward(self):
        return self.w


# TODO: Add progressive growing
# TODO: Add sin model
# TODO: Add Dall-e decoder
# TODO: Add stylegan

