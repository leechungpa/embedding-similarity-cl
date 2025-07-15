import torch
import torch.nn.functional as F
from solo.utils.misc import gather, get_rank


def dcl_loss_func(
    z: torch.Tensor, indexes: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """
    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).

    Return:
        torch.Tensor: DHEL loss.
    """
    z = F.normalize(z, dim=-1)
    gathered_z = gather(z)

    sim = torch.exp(torch.einsum("if, jf -> ij", z, gathered_z) / temperature)

    gathered_indexes = gather(indexes)
    indexes = indexes.unsqueeze(0)
    gathered_indexes = gathered_indexes.unsqueeze(0)

    # positives
    pos_mask = indexes.t() == gathered_indexes
    pos_mask[:, z.size(0) * get_rank() :].fill_diagonal_(0)

    # negatives
    neg_mask = indexes.t() != gathered_indexes

    pos = torch.sum(sim * pos_mask, dim=1)
    neg = torch.sum(sim * neg_mask, 1)
    loss = -torch.mean(torch.log(pos / neg))
    return loss
