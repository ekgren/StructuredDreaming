import numpy as np
import torch


def training_loop(
        random_seed=0, #Global random seed.
):
    # Initialize.
    device = torch.device('cuda')
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
