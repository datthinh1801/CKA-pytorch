import torch


def batched_hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Computes the Hilbert-Schmidt Independence Criterion (HSIC) in a batched manner.

    HSIC is a measure of statistical independence between two random variables.
    This function calculates the HSIC between two kernel matrices, K and L, for a batch of data.

    Args:
        K: A batch of kernel matrices of shape (B, N, N), where B is the batch size
           and N is the number of samples.
        L: A batch of kernel matrices of the same shape as K.

    Returns:
        A tensor of shape (B,) containing the HSIC value for each item in the batch.
    """
    assert K.size() == L.size(), "Kernel matrices must have the same dimensions."
    assert K.dim() == 3, "Input tensors must be 3-dimensional (B, N, N)."

    K = K.clone()
    L = L.clone()
    n = K.size(1)

    # Zero out the diagonals
    K.diagonal(dim1=-1, dim2=-2).fill_(0)
    L.diagonal(dim1=-1, dim2=-2).fill_(0)

    # HSIC calculation
    KL = torch.bmm(K, L)
    trace_KL = KL.diagonal(dim1=-1, dim2=-2).sum(-1).unsqueeze(-1).unsqueeze(-1)

    sum_K = K.sum((-1, -2), keepdim=True)
    sum_L = L.sum((-1, -2), keepdim=True)
    middle_term = sum_K * sum_L / ((n - 1) * (n - 2))

    sum_KL = KL.sum((-1, -2), keepdim=True)
    right_term = 2 * sum_KL / (n - 2)

    hsic = (trace_KL + middle_term - right_term) / (n * (n - 3))

    return hsic.squeeze(-1).squeeze(-1)
