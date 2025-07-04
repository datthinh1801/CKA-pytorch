import torch


def debiased_gram(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the Gram matrix of the input tensor `x`.

    The Gram matrix `G` is a square matrix where each element `G_ij` represents
    the inner product of the i-th and j-th feature vectors from the input `x`.
    Specifically, `G = X @ X.T`.
    This function is useful for capturing the relationships and similarities
    between different samples in a batch based on their feature representations.

    Args:
        x: A `torch.Tensor` of shape `(N, D)`, where `N` is the number of samples
           (e.g., batch size) and `D` is the feature dimension.

    Returns:
        A `torch.Tensor` representing the Gram matrix of shape `(N, N)`.
    """
    return x.matmul(x.t())


def linear_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the Gram (kernel) matrix for a linear kernel.

    The linear kernel is defined as `K(x, y) = x^T y`. For a matrix `X` where each row is a sample,
    the Gram matrix is `X @ X.T`.
    This function is adapted from the implementation by Kornblith et al.
    (https://github.com/google-research/google-research/tree/master/representation_similarity).

    Args:
        x: A `torch.Tensor` of shape `(n_samples, n_features)`.

    Returns:
        A `torch.Tensor` representing the Gram matrix of shape `(n_samples, n_samples)`.
    """
    return torch.mm(x, x.T)


def rbf_kernel(x: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """
    Computes the Gram (kernel) matrix for a Radial Basis Function (RBF) kernel.

    The RBF kernel, also known as the Gaussian kernel, is a popular choice for measuring
    similarity between data points. It is defined as `K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))`,
    where `sigma` is the bandwidth parameter.
    This implementation uses a `threshold` to scale the median Euclidean distance for `sigma`.
    This function is adapted from the implementation by Kornblith et al.
    (https://github.com/google-research/google-research/tree/master/representation_similarity).

    Args:
        x: A `torch.Tensor` of shape `(n_samples, n_features)`.
        threshold: A float value that determines the bandwidth of the RBF kernel.
                   It is used as a fraction of the median Euclidean distance between samples.
                   A larger `threshold` results in a wider kernel (less sensitive to small differences).
                   Defaults to 1.0.

    Returns:
        A `torch.Tensor` representing the Gram matrix of shape `(n_samples, n_samples)`.
    """
    dot_products = torch.mm(x, x.T)
    sq_norms = torch.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = torch.median(sq_distances)
    return torch.exp(-sq_distances / (2 * threshold**2 * sq_median_distance))


def center_gram_matrix(
    gram_matrix: torch.Tensor, unbiased: bool = False
) -> torch.Tensor:
    """
    Centers a given Gram matrix.

    Gram matrix centering is a crucial step in CKA calculation, as it ensures that
    the kernel values are centered around zero, which is necessary for the HSIC
    calculation to be meaningful. This function can perform either a biased or
    unbiased centering.

    This function is adapted from the implementation by Kornblith et al.
    (https://github.com/google-research/google-research/tree/master/representation_similarity).

    Args:
        gram_matrix: A `torch.Tensor` representing the Gram matrix of shape `(n, n)`,
                       where `n` is the number of samples.
        unbiased: A boolean indicating whether to use the unbiased version of the centering.
                  Defaults to `False`.

    Returns:
        A `torch.Tensor` representing the centered version of the given Gram matrix,
        with the same shape as the input `gram_matrix`.

    Raises:
        ValueError: If the input `gram_matrix` is not symmetric.
    """
    if not torch.allclose(gram_matrix, gram_matrix.T):
        raise ValueError("The given matrix must be symmetric.")

    gram_matrix = gram_matrix.detach().clone()
    if unbiased:
        n = gram_matrix.shape[0]
        gram_matrix.fill_diagonal_(0)
        means = torch.sum(gram_matrix, dim=0, dtype=torch.float64) / (n - 2)
        means -= torch.sum(means) / (2 * (n - 1))
        gram_matrix -= means[:, None]
        gram_matrix -= means[None, :]
        gram_matrix.fill_diagonal_(0)
    else:
        means = torch.mean(gram_matrix, dim=0, dtype=torch.float64)
        means -= torch.mean(means) / 2
        gram_matrix -= means[:, None]
        gram_matrix -= means[None, :]

    return gram_matrix
