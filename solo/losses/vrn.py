import torch
import torch.nn.functional as F

from solo.utils.misc import gather


def vrn_loss_func(
    z1: torch.Tensor, z2: torch.Tensor,
    index_or_target: torch.Tensor,
    delta: float,
    p: int = 2,
) -> torch.Tensor:
    """
    Computes the VRN (Variance-Reduction for Negative-pair similarity) loss term.
    
    Args:
        z1 (torch.Tensor): Normalized embedding vectors with shape (B, proj_output_dim).
        z2 (torch.Tensor): Normalized embedding vectors with shape (B, proj_output_dim).
        index_or_target (torch.Tensor): Indexs or target labels in the batch.
        delta (float): A predefined cosine similarity for negative pairs.
        p (int): Exponent value for the power operation in the loss formulation.
    
    Returns:
        torch.Tensor: The computed VRN loss value.
    """
    z1 = F.normalize(z1, dim=1) # ( B, proj_output_dim )
    z2 = F.normalize(z2, dim=1) # ( B, proj_output_dim )

    gathered_z2 = gather(z2)
    gathered_index_or_target = gather(index_or_target)

    similarity = torch.einsum("id, jd -> ij", z1, gathered_z2)

    index_or_target = index_or_target.unsqueeze(0)
    gathered_index_or_target = gathered_index_or_target.unsqueeze(0)

    neg_mask = index_or_target.t() != gathered_index_or_target

    return (similarity[neg_mask] - delta).pow(p).mean()