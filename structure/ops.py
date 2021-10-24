import torch

# From the paper "CLOOB: Modern Hopfield Networks with InfoLOOB Outperform CLIP"
# Arxiv link: https://arxiv.org/abs/2110.11316
# Code: https://github.com/ml-jku/cloob/
def infoLOOB_loss(x, y, i, inv_tau):
    tau = 1 / inv_tau
    k = x @ y.T / tau
    positives = -torch.mean(torch.sum(k * i, dim=1))

    # For logsumexp the zero entries must be equal to a very large negative number
    large_neg = -1000.0
    arg_lse = k * torch.logical_not(i) + i * large_neg
    negatives = torch.mean(torch.logsumexp(arg_lse, dim=1))

    return tau * (positives + negatives)