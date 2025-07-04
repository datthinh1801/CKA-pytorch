from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm

from cka_pytorch.hook_manager import HookManager
from cka_pytorch.hsic import hsic1
from cka_pytorch.metrics import AccumTensor
from cka_pytorch.plot import plot_cka

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class CKACalculator:
    """
    A class to calculate the Centered Kernel Alignment (CKA) matrix between two PyTorch models.

    CKA is a similarity metric that measures the similarity between the representations
    (activations) of two neural networks. It is particularly useful for comparing
    different models or different layers within the same model.
    """

    def __init__(
        self,
        model1: nn.Module,
        model2: nn.Module,
        model1_layers: List[str],
        model2_layers: List[str] | None = None,
        model1_name: str = "Model 1",
        model2_name: str = "Model 2",
        batched_feature_size: int = 64,
        device: Optional[torch.device] = None,
        verbose: bool = True,
    ) -> None:
        """
        Initializes the CKACalculator with two models and their respective layers for CKA computation.

        Args:
            model1: The first PyTorch model. Its activations from specified layers will be used.
            model2: The second PyTorch model. Its activations from specified layers will be used.
            model1_layers: A list of strings, where each string is the name of a layer in `model1`
                           whose activations are to be extracted. These names should correspond to
                           names returned by `model1.named_modules()`.
            model2_layers: An optional list of strings, similar to `model1_layers` but for `model2`.
                           If `None`, `model1_layers` will be used for `model2` as well.
            model1_name: An optional string representing the name of `model1`, used for plotting.
                         Defaults to "Model 1".
            model2_name: An optional string representing the name of `model2`, used for plotting.
                         Defaults to "Model 2".
            batched_feature_size: An integer specifying the number of layers to process in a single batch
                                 when calculating HSIC. This is useful for memory efficiency, especially
                                 when dealing with large models or many layers. Defaults to 64.
            device: An optional `torch.device` to perform computations on (e.g., `torch.device("cuda")`
                    or `torch.device("cpu")`). If `None`, the device of `model1`'s parameters will be used.
            verbose: A boolean indicating whether to print progress bars during CKA calculation.
                     Defaults to `True`.
        """
        self.model1 = model1
        self.model2 = model2
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.batched_feature_size = batched_feature_size
        self.device = device or next(model1.parameters()).device
        self.verbose = verbose

        self.model1.eval()
        self.model2.eval()

        self.hook_manager1 = HookManager(model1, model1_layers)
        self.hook_manager2 = HookManager(
            model2, model2_layers if model2_layers else model1_layers
        )

        self.num_layers_x = len(self.hook_manager1.module_names)
        self.num_layers_y = len(self.hook_manager2.module_names)
        self.num_elements = self.num_layers_x * self.num_layers_y

        self.hsic_matrix = AccumTensor(
            torch.zeros(self.num_layers_y, self.num_layers_x, device=self.device)
        )
        self.self_hsic_x = AccumTensor(
            torch.zeros(self.num_layers_x, device=self.device)
        )
        self.self_hsic_y = AccumTensor(
            torch.zeros(self.num_layers_y, device=self.device)
        )

    @torch.no_grad()
    def calculate_cka_matrix(
        self,
        dataloader: DataLoader,
        num_epochs: int = 10,
        epsilon: float = 1e-4,
    ) -> torch.Tensor:
        """
        Calculates the CKA matrix by processing data from the provided DataLoader.

        The CKA matrix is computed by accumulating Hilbert-Schmidt Independence Criterion (HSIC)
        values over multiple batches and epochs. The final CKA value for each layer pair
        is then derived from these accumulated HSIC values.

        Args:
            dataloader: A `torch.utils.data.DataLoader` providing the input data.
                        It's recommended that the DataLoader does not drop the last batch
                        (`drop_last=False`) to ensure all samples contribute to the CKA calculation.
            num_epochs: The number of times to iterate over the entire `dataloader`.
                        Increasing this can lead to more stable CKA estimates, especially with noisy data.
                        Defaults to 10.
            epsilon: A small float value added to the denominator during the final CKA calculation
                     to prevent division by zero in cases where self-HSIC values might be very small.
                     Defaults to 1e-4.

        Returns:
            A `torch.Tensor` representing the CKA matrix. The dimensions of the matrix will be
            (number of `model1_layers`, number of `model2_layers`). Each element `(i, j)`
            in the matrix represents the CKA similarity between the i-th layer of `model1`
            and the j-th layer of `model2`.
        """
        for epoch in range(num_epochs):
            loader = tqdm(
                dataloader,
                desc=f"Calculate CKA matrix (Epoch {epoch+1}/{num_epochs})",
                disable=not self.verbose,
            )
            for x, _ in loader:
                self._process_batch(x.to(self.device))

        return self._compute_final_cka(epsilon)

    def _process_batch(self, x: torch.Tensor) -> None:
        """
        Processes a single batch of input data to extract features and update the HSIC accumulators.

        This method performs a forward pass through both models with the given batch `x`,
        collects the activations from the specified layers using the `HookManager`,
        and then calls `_update_hsic_matrices` to update the accumulated HSIC values.
        Finally, it clears the collected features to prepare for the next batch.

        Args:
            x: A `torch.Tensor` representing a batch of input data. This tensor is moved
               to the appropriate device (CPU/GPU) before processing.
        """
        _ = self.model1(x)
        _ = self.model2(x)

        features1 = [
            self.hook_manager1.features[layer]
            for layer in self.hook_manager1.module_names
        ]
        features2 = [
            self.hook_manager2.features[layer]
            for layer in self.hook_manager2.module_names
        ]

        self._update_hsic_matrices(features1, features2)

        self.hook_manager1.clear_features()
        self.hook_manager2.clear_features()

    def _update_hsic_matrices(
        self,
        features1: List[torch.Tensor],
        features2: List[torch.Tensor],
    ) -> None:
        """
        Calculates and updates the self-HSIC and cross-HSIC matrices in a mini-batched manner.

        This method takes the extracted features from both models for the current batch,
        computes their respective kernel matrices, and then calculates the self-HSIC
        (HSIC(X, X) and HSIC(Y, Y)) and cross-HSIC (HSIC(X, Y)) values.
        These values are then accumulated into `self.hsic_matrix`, `self.self_hsic_x`,
        and `self.self_hsic_y` using the `AccumTensor` metric.

        Args:
            features1: A list of `torch.Tensor`s, where each tensor represents the activations
                       from a layer of `model1` for the current batch.
            features2: A list of `torch.Tensor`s, where each tensor represents the activations
                       from a layer of `model2` for the current batch.
        """
        hsic_x = torch.zeros(1, self.num_layers_x, device=self.device)
        hsic_y = torch.zeros(self.num_layers_y, 1, device=self.device)
        hsic_matrix = torch.zeros(self.num_elements, device=self.device)

        # Self-HSIC
        for start_idx in range(0, self.num_layers_x, self.batched_feature_size):
            end_idx = min(start_idx + self.batched_feature_size, self.num_layers_x)
            gram_x = torch.stack(features1[start_idx:end_idx], dim=0)
            hsic_x[0, start_idx:end_idx] += hsic1(gram_x, gram_x)
        self.self_hsic_x.update(hsic_x)

        for start_idx in range(0, self.num_layers_y, self.batched_feature_size):
            end_idx = min(start_idx + self.batched_feature_size, self.num_layers_y)
            gram_y = torch.stack(features2[start_idx:end_idx], dim=0)
            hsic_y[start_idx:end_idx, 0] += hsic1(gram_y, gram_y)
        self.self_hsic_y.update(hsic_y)

        # Cross-HSIC
        for start_idx in range(0, self.num_elements, self.batched_feature_size):
            end_idx = min(start_idx + self.batched_feature_size, self.num_elements)
            gram_x = torch.stack(
                [features1[i % self.num_layers_x] for i in range(start_idx, end_idx)],
                dim=0,
            )
            gram_y = torch.stack(
                [features2[i // self.num_layers_x] for i in range(start_idx, end_idx)],
                dim=0,
            )
            hsic_matrix[start_idx:end_idx] += hsic1(gram_x, gram_y)
        self.hsic_matrix.update(hsic_matrix)

    def _compute_final_cka(self, epsilon: float) -> torch.Tensor:
        """
        Computes the final CKA matrix from the accumulated HSIC values.

        This method is called after all batches and epochs have been processed.
        It retrieves the final accumulated cross-HSIC matrix (`hsic_matrix`)
        and the accumulated self-HSIC vectors for `model1` (`self_hsic_x`)
        and `model2` (`self_hsic_y`).
        The CKA value for each layer pair `(i, j)` is then calculated as:
        CKA(i, j) = HSIC(X_i, Y_j) / sqrt(HSIC(X_i, X_i) * HSIC(Y_j, Y_j))
        where X_i and Y_j are the activations of layer i from model1 and layer j from model2, respectively.

        Args:
            epsilon: A small float value added to the denominator to prevent division by zero.

        Returns:
            A `torch.Tensor` representing the final CKA matrix.
        """
        hsic_matrix = self.hsic_matrix.compute()
        self_hsic_x = self.self_hsic_x.compute()
        self_hsic_y = self.self_hsic_y.compute()

        cka_matrix = hsic_matrix.reshape(
            self.num_layers_y, self.num_layers_x
        ) / torch.sqrt(self_hsic_x * self_hsic_y + epsilon)
        return cka_matrix

    def plot_cka_matrix(
        self,
        cka_matrix: torch.Tensor,
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
        plot_cka(
            cka_matrix=cka_matrix,
            model1_layers=self.hook_manager1.module_names,
            model2_layers=self.hook_manager2.module_names,
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
        Resets the accumulators for a new CKA calculation.

        This method clears the accumulated HSIC values and resets the hook managers
        to prepare for a new CKA calculation.
        """
        self.hsic_matrix.reset()
        self.self_hsic_x.reset()
        self.self_hsic_y.reset()

        self.hook_manager1.clear_all()
        self.hook_manager2.clear_all()
