import torch
from torchmetrics import Metric


class AccumTensor(Metric):
    """
    A `torchmetrics` Metric to accumulate tensors over multiple updates.

    This metric is designed to sum tensors element-wise over several steps.
    It is useful in distributed training scenarios, as it supports tensor
    accumulation across different devices or processes.

    The accumulated tensor retains the shape of the initial `default_value`.
    """

    def __init__(self, default_value: torch.Tensor):
        """
        Initializes the AccumTensor metric.

        Args:
            default_value: A tensor with the desired shape, used to initialize
                           the accumulated value.
        """
        super().__init__()
        self.add_state("val", default=default_value, dist_reduce_fx="sum")

    def update(self, input_tensor: torch.Tensor) -> None:
        """
        Updates the accumulated tensor by adding the input tensor.

        Args:
            input_tensor: The tensor to add to the accumulated value.
        """
        self.val += input_tensor

    def compute(self) -> torch.Tensor:
        """
        Returns the final accumulated tensor.

        Returns:
            The tensor holding the sum of all updated tensors.
        """
        return self.val
