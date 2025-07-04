import torch
from torchmetrics import Metric


class AccumTensor(Metric):
    """
    A `torchmetrics` Metric designed to accumulate `torch.Tensor` values over multiple updates.

    This metric is particularly useful for scenarios where tensors need to be summed
    element-wise across different mini-batches or distributed training processes.
    It leverages `torchmetrics`'s state management and distributed reduction capabilities.

    The accumulated tensor maintains the shape and device of the `default_value`
    provided during initialization.
    """

    def __init__(self, default_value: torch.Tensor):
        """
        Initializes a new instance of the `AccumTensor` metric.

        Args:
            default_value: A `torch.Tensor` that serves as the initial value and defines
                           the shape and data type of the tensor to be accumulated.
                           This tensor will be used to initialize the internal state `self.val`.
        """
        super().__init__()
        # Register 'val' as a persistent state. 'dist_reduce_fx="sum"' ensures that
        # in distributed settings, the 'val' from all processes are summed together.
        self.add_state("val", default=default_value, dist_reduce_fx="sum")

    def update(self, input_tensor: torch.Tensor) -> None:
        """
        Updates the accumulated tensor by adding the `input_tensor` element-wise.

        This method is called for each new tensor that needs to be accumulated.
        The `input_tensor` is added to the current accumulated value stored in `self.val`.

        Args:
            input_tensor: The `torch.Tensor` to be added to the accumulated value.
                          It must have the same shape and data type as the `default_value`
                          provided during initialization.
        """
        self.val += input_tensor

    def compute(self) -> torch.Tensor:
        """
        Returns the final accumulated tensor.

        Returns:
            The tensor holding the sum of all updated tensors.
        """
        return self.val
