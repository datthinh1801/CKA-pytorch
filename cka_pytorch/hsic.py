import torch


def hsic1(gram_x: torch.Tensor, gram_y: torch.Tensor) -> torch.Tensor:
    """
    Computes the batched version of the Hilbert-Schmidt Independence Criterion (HSIC) on Gram matrices.

    This function is designed to work with mini-batches of data, where `gram_x` and `gram_y`
    are collections of Gram matrices, one for each sample in the batch.
    It calculates an unbiased estimator of HSIC for each pair of Gram matrices in the batch.

    Args:
        gram_x: A `torch.Tensor` representing a batch of Gram matrices for the first set of features (X).
                Expected shape: `(batch_size, n, n)`, where `batch_size` is the number of samples
                in the mini-batch, and `n` is the number of data points (e.g., features or neurons).
        gram_y: A `torch.Tensor` representing a batch of Gram matrices for the second set of features (Y).
                Expected shape: `(batch_size, n, n)`, same dimensions as `gram_x`.

    Returns:
        A `torch.Tensor` of shape `(batch_size,)` containing the unbiased HSIC value for each
        pair of Gram matrices in the batch.

    Raises:
        ValueError: If `gram_x` and `gram_y` do not have exactly three dimensions or if their
                    shapes do not match.
    """
    if len(gram_x.size()) != 3 or gram_x.size() != gram_y.size():
        raise ValueError("Invalid size for one of the two input tensors.")

    n = gram_x.shape[-1]
    gram_x = gram_x.clone()
    gram_y = gram_y.clone()

    # Fill the diagonal of each matrix with 0
    gram_x.diagonal(dim1=-1, dim2=-2).fill_(0)
    gram_y.diagonal(dim1=-1, dim2=-2).fill_(0)

    # Compute the product between gram_x and gram_y
    kl = torch.bmm(gram_x, gram_y)

    # Compute the trace (sum of the elements on the diagonal) of the previous product, i.e.: the left term
    trace_kl = kl.diagonal(dim1=-1, dim2=-2).sum(-1).unsqueeze(-1).unsqueeze(-1)

    # Compute the middle term
    sum_gram_x = gram_x.sum((-1, -2), keepdim=True)
    sum_gram_y = gram_y.sum((-1, -2), keepdim=True)
    middle_term = sum_gram_x * sum_gram_y / ((n - 1) * (n - 2))

    # Compute the right term
    sum_kl = kl.sum((-1, -2), keepdim=True)
    right_term = 2 * sum_kl / (n - 2)

    # Put all together to compute the main term
    hsic = (trace_kl + middle_term - right_term) / (n**2 - 3 * n)

    return hsic.squeeze(-1).squeeze(-1)
