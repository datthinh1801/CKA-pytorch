from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm

from cka_pytorch.hook_manager import _HOOK_LAYER_TYPES, HookManager
from cka_pytorch.hsic import batched_hsic
from cka_pytorch.metrics import AccumTensor
from cka_pytorch.plot import plot_cka

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class CKACalculator:
    """
    A class to calculate the Centered Kernel Alignment (CKA) matrix between two models.

    CKA is a similarity metric that can be used to compare the representations learned by two neural networks.
    This class facilitates the process by managing hooks, extracting intermediate features, and computing the CKA matrix.

    The CKA calculation involves the following steps:
    1.  Registering hooks to the specified layers of both models.
    2.  Passing data through the models to capture intermediate feature representations.
    3.  Computing the Gram matrix for each layer's features.
    4.  Calculating the Hilbert-Schmidt Independence Criterion (HSIC) between the Gram matrices.
    5.  Normalizing the HSIC matrix to obtain the final CKA matrix.
    """

    def __init__(
        self,
        model1: nn.Module,
        model2: nn.Module,
        dataloader: DataLoader,
        model1_name: str = "Model 1",
        model2_name: str = "Model 2",
        hook_fn: Optional[Union[str, Callable]] = None,
        hook_layer_types: Tuple[Type[nn.Module], ...] = _HOOK_LAYER_TYPES,
        num_epochs: int = 10,
        group_size: int = 512,
        epsilon: float = 1e-4,
        is_main_process: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initializes the CKACalculator.

        Args:
            model1: The first model to evaluate. Must be an instance of `nn.Module`.
            model2: The second model to evaluate. Must be an instance of `nn.Module`.
            dataloader: A PyTorch `DataLoader` for loading the dataset.
            hook_fn: The function or hook name to use for feature extraction. Can be 'flatten' or 'avgpool'.
                     Defaults to 'flatten'.
            hook_layer_types: A tuple of layer types to which hooks will be attached.
            num_epochs: The number of epochs to run the CKA calculation over.
            group_size: The batch size for processing layers to optimize memory usage.
            epsilon: A small constant added to the denominator for numerical stability.
            is_main_process: A flag to indicate if the current process is the main one, used for distributed training.
            device: The device on which to perform computations. If `None`, it is inferred from the model's parameters.
        """
        self.model1 = model1
        self.model2 = model2
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.group_size = group_size
        self.epsilon = epsilon
        self.is_main_process = is_main_process
        self.device = device or next(model1.parameters()).device

        self.model1.eval()
        self.model2.eval()
        self.hook_manager1 = HookManager(model1, hook_fn, hook_layer_types)
        self.hook_manager2 = HookManager(model2, hook_fn, hook_layer_types)

        self.module_names_x: Optional[List[str]] = None
        self.module_names_y: Optional[List[str]] = None
        self.num_layers_x: Optional[int] = None
        self.num_layers_y: Optional[int] = None
        self.num_elements: Optional[int] = None

        self.cka_matrix: Optional[torch.Tensor] = None
        self.hsic_matrix: Optional[AccumTensor] = None
        self.self_hsic_x: Optional[AccumTensor] = None
        self.self_hsic_y: Optional[AccumTensor] = None

    @torch.no_grad()
    def calculate_cka_matrix(self) -> torch.Tensor:
        """
        Calculates and returns the CKA matrix between the two models.

        This method orchestrates the entire CKA calculation process, including running the epochs,
        processing batches, and computing the final matrix.

        Returns:
            The computed CKA matrix, with dimensions (num_layers_y, num_layers_x).
        """
        self._run_epochs()
        return self._compute_final_cka()

    def _run_epochs(self) -> None:
        """
        Runs the calculation for the specified number of epochs, accumulating HSIC values.
        """
        for epoch in range(self.num_epochs):
            loader = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch+1}/{self.num_epochs}",
                disable=not self.is_main_process,
            )
            for x, _ in loader:
                self._process_batch(x.to(self.device))

    def _process_batch(self, x: torch.Tensor) -> None:
        """
        Processes a single batch of data to update the HSIC matrices.

        Args:
            x: A batch of input data.
        """
        _ = self.model1(x)
        _ = self.model2(x)
        features1, features2 = self._extract_features()

        if self.num_layers_x is None:
            self._initialize_metrics(features1, features2)

        self._update_hsic_matrices(features1, features2)
        self.hook_manager1.clear_features()
        self.hook_manager2.clear_features()

    def _extract_features(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Extracts intermediate features from the hook managers of both models.

        Returns:
            A tuple containing two lists of feature tensors for model 1 and model 2.
        """
        return self.hook_manager1.features, self.hook_manager2.features

    def _initialize_metrics(self, features1: list, features2: list) -> None:
        """
        Initializes the metrics and related variables based on the first batch of features.

        Args:
            features1: A list of feature tensors from the first model.
            features2: A list of feature tensors from the second model.
        """
        self.num_layers_x = len(features1)
        self.num_layers_y = len(features2)
        self.module_names_x = self.hook_manager1.module_names
        self.module_names_y = self.hook_manager2.module_names
        self.num_elements = self.num_layers_y * self.num_layers_x

        self.hsic_matrix = AccumTensor(
            torch.zeros(self.num_elements, device=self.device)
        )
        self.self_hsic_x = AccumTensor(
            torch.zeros(1, self.num_layers_x, device=self.device)
        )
        self.self_hsic_y = AccumTensor(
            torch.zeros(self.num_layers_y, 1, device=self.device)
        )

    def _update_hsic_matrices(self, features1: list, features2: list) -> None:
        """
        Calculates and updates the self-HSIC and cross-HSIC matrices.

        Args:
            features1: A list of feature tensors from the first model.
            features2: A list of feature tensors from the second model.
        """
        assert self.num_layers_x is not None and self.num_layers_y is not None
        assert self.self_hsic_x is not None and self.self_hsic_y is not None
        assert self.num_elements is not None
        assert self.hsic_matrix is not None

        hsic_x = torch.zeros(1, self.num_layers_x, device=self.device)
        hsic_y = torch.zeros(self.num_layers_y, 1, device=self.device)
        hsic_matrix = torch.zeros(self.num_elements, device=self.device)

        # Self-HSIC for model 1
        for start_idx in range(0, self.num_layers_x, self.group_size):
            end_idx = min(start_idx + self.group_size, self.num_layers_x)
            K = torch.stack(features1[start_idx:end_idx], dim=0)
            hsic_x[0, start_idx:end_idx] += batched_hsic(K, K) * self.epsilon
        self.self_hsic_x.update(hsic_x)

        # Self-HSIC for model 2
        for start_idx in range(0, self.num_layers_y, self.group_size):
            end_idx = min(start_idx + self.group_size, self.num_layers_y)
            L = torch.stack(features2[start_idx:end_idx], dim=0)
            hsic_y[start_idx:end_idx, 0] += batched_hsic(L, L) * self.epsilon
        self.self_hsic_y.update(hsic_y)

        # Cross-HSIC
        for start_idx in range(0, self.num_elements, self.group_size):
            end_idx = min(start_idx + self.group_size, self.num_elements)
            K = torch.stack(
                [features1[i % self.num_layers_x] for i in range(start_idx, end_idx)],
                dim=0,
            )
            L = torch.stack(
                [features2[j // self.num_layers_x] for j in range(start_idx, end_idx)],
                dim=0,
            )
            hsic_matrix[start_idx:end_idx] += batched_hsic(K, L) * self.epsilon
        self.hsic_matrix.update(hsic_matrix)

    def _compute_final_cka(self) -> torch.Tensor:
        """
        Computes the final CKA matrix from the accumulated HSIC values.

        Returns:
            The normalized CKA matrix.
        """
        assert self.hsic_matrix is not None
        assert self.self_hsic_x is not None
        assert self.self_hsic_y is not None
        assert self.num_layers_x is not None and self.num_layers_y is not None

        hsic_matrix = self.hsic_matrix.compute()
        hsic_x = self.self_hsic_x.compute()
        hsic_y = self.self_hsic_y.compute()

        self.cka_matrix = hsic_matrix.reshape(
            self.num_layers_y, self.num_layers_x
        ) / torch.sqrt(hsic_x * hsic_y)
        return self.cka_matrix

    def plot_cka_matrix(
        self,
        save_path: str | None = None,
        title: str | None = None,
        vmin: float = 0.0,
        vmax: float = 1.0,
        cmap: str = "magma",
        show_ticks_labels: bool = True,
        short_tick_labels_splits: int | None = None,
        use_tight_layout: bool = True,
        show_annotations: bool = True,
        show_img: bool = True,
        show_half_heatmap: bool = False,
        invert_y_axis: bool = True,
        title_font_size: int = 14,
        axis_font_size: int = 12,
        tick_font_size: int = 10,
        figsize: tuple[int, int] = (10, 10),
        dpi: int = 300,
    ) -> None:
        """Plot the CKA matrix.

        Args:
            save_path (str | None): Where to save the plot. If None, the plot will not be saved.
            title (str | None): The plot title. If None, a default title will be used.
            vmin (float): Minimum value for the colormap.
            vmax (float): Maximum value for the colormap.
            cmap (str): The name of the colormap to use.
            show_ticks_labels (bool): Whether to show the tick labels.
            short_tick_labels_splits (int | None): If not None, shorten tick labels.
            use_tight_layout (bool): Whether to use a tight layout.
            show_annotations (bool): Whether to show annotations on the heatmap.
            show_img (bool): Whether to show the plot.
            show_half_heatmap (bool): Whether to show only half of the heatmap.
            invert_y_axis (bool): Whether to invert the y-axis.
        """
        if self.cka_matrix is None:
            raise ValueError(
                "CKA matrix has not been calculated yet. Call `calculate_cka_matrix` first."
            )

        plot_cka(
            cka_matrix=self.cka_matrix,
            model1_layers=self.module_names_x,  # type: ignore
            model2_layers=self.module_names_y,  # type: ignore
            model1_name=self.model1_name,
            model2_name=self.model2_name,
            save_path=save_path,
            title=title,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            show_ticks_labels=show_ticks_labels,
            short_tick_labels_splits=short_tick_labels_splits,
            use_tight_layout=use_tight_layout,
            show_annotations=show_annotations,
            show_img=show_img,
            show_half_heatmap=show_half_heatmap,
            invert_y_axis=invert_y_axis,
            title_font_size=title_font_size,
            axis_font_size=axis_font_size,
            tick_font_size=tick_font_size,
            figsize=figsize,
            dpi=dpi,
        )

    def reset(self) -> None:
        """
        Resets the calculator by clearing all accumulated metrics and hooks.

        This method is useful when you want to reuse the calculator for a new comparison.
        """
        self.cka_matrix = None
        self.hsic_matrix = None
        self.self_hsic_x = None
        self.self_hsic_y = None
        self.hook_manager1.clear_all()
        self.hook_manager2.clear_all()
