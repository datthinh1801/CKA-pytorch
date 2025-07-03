import torch


def gram(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the Gram matrix of the input tensor.

    The Gram matrix is a square matrix of inner products, where G_ij = v_i^T v_j.
    In this context, it is used to capture the relationships between feature vectors
    in a set of samples.

    Args:
        x: A tensor of shape (N, D), where N is the number of samples (batch size)
           and D is the feature dimension.

    Returns:
        The Gram matrix of shape (N, N).
    """
    return x.matmul(x.t())
