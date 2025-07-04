import torch


def hsic0(gram_x: torch.Tensor, gram_y: torch.Tensor) -> torch.Tensor:
    """
    Computes the Hilbert-Schmidt Independence Criterion (HSIC) for two given Gram matrices.

    HSIC is a measure of dependence between two random variables. In the context of CKA,
    it quantifies the statistical dependence between the features of two layers.
    This implementation uses the unbiased estimator of HSIC.

    Args:
        gram_x: A `torch.Tensor` representing the Gram matrix of the first set of features (X).
                Expected shape: `(n, n)`, where `n` is the number of samples.
        gram_y: A `torch.Tensor` representing the Gram matrix of the second set of features (Y).
                Expected shape: `(n, n)`, where `n` is the number of samples.

    Returns:
        A `torch.Tensor` (scalar) representing the HSIC value.

    Raises:
        ValueError: If either `gram_x` or `gram_y` is not a symmetric matrix.
    """
    if not torch.allclose(gram_x, gram_x.T) and not torch.allclose(gram_y, gram_y.T):
        raise ValueError("The given matrices must be symmetric.")

    # Build the identity matrix
    n = gram_x.shape[0]
    identity = torch.eye(n, n, dtype=gram_x.dtype, device=gram_x.device)

    # Build the centering matrix
    h = identity - torch.ones(n, n, dtype=gram_x.dtype, device=gram_x.device) / n

    # Compute k * h and l * h
    kh = torch.mm(gram_x, h)
    lh = torch.mm(gram_y, h)

    # Compute the trace of the product kh * lh
    trace = torch.trace(kh.mm(lh))
    return trace / (n - 1) ** 2


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

    # Compute the product between k (i.e.: gram_x) and l (i.e.: gram_y)
    kl = torch.bmm(gram_x, gram_y)

    # Compute the trace (sum of the elements on the diagonal) of the previous product, i.e.: the left term
    trace_kl = kl.diagonal(dim1=-1, dim2=-2).sum(-1).unsqueeze(-1).unsqueeze(-1)

    # Compute the middle term
    middle_term = gram_x.sum((-1, -2), keepdim=True) * gram_y.sum(
        (-1, -2), keepdim=True
    )
    middle_term /= (n - 1) * (n - 2)

    # Compute the right term
    right_term = kl.sum((-1, -2), keepdim=True)
    right_term *= 2 / (n - 2)

    # Put all together to compute the main term
    main_term = trace_kl + middle_term - right_term

    # Compute the hsic values
    out = main_term / (n**2 - 3 * n)
    return out.squeeze(-1).squeeze(-1)
