import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import torch


def plot_cka(
    cka_matrix: torch.Tensor,
    model1_layers: list[str],
    model2_layers: list[str],
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
    dirpath: str | None = None,
    filepath: str | None = None,
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
    """
    Plots a Centered Kernel Alignment (CKA) matrix as a heatmap.

    This function visualizes the CKA similarity between layers of two models.
    The heatmap displays the CKA values, providing insights into the functional
    similarity of different layers across models.

    Args:
        cka_matrix: A `torch.Tensor` representing the CKA matrix. Expected shape is (N, M),
                    where N is the number of layers in model1 and M is the number of layers in model2.
                    Values typically range from 0 (no similarity) to 1 (high similarity).
        model1_layers: A list of strings, where each string is the name of a layer from the first model.
                       These names will be used as labels for the y-axis of the heatmap.
        model2_layers: A list of strings, where each string is the name of a layer from the second model.
                       These names will be used as labels for the x-axis of the heatmap.
        model1_name: An optional string representing the name of the first model. Used in the plot title and y-axis label.
                     Defaults to "Model 1".
        model2_name: An optional string representing the name of the second model. Used in the plot title and x-axis label.
                     Defaults to "Model 2".
        dirpath: An optional string specifying the directory path where the plot should be saved.
                   If `None`, the plot will be displayed but not saved. The filename will be generated
                   based on the title.
        filepath: An optional string specifying the full file path where the plot should be saved.
                  If provided, this will override `dirpath`. The filename will be generated based on the
                  title. If `None`, the plot will not be saved.
        title: An optional string to be used as the main title of the plot. If `None`,
               a default title like "Model 1 vs Model 2" will be generated.
        vmin: The minimum value for the colormap range. Values below `vmin` will be clipped.
              Defaults to 0.0.
        vmax: The maximum value for the colormap range. Values above `vmax` will be clipped.
              Defaults to 1.0.
        cmap: The name of the Matplotlib colormap to use for the heatmap. Defaults to "magma".
        show_ticks_labels: A boolean indicating whether to display the layer names as tick labels on the axes.
                           Defaults to `True`.
        short_tick_labels_splits: An optional integer. If provided and `show_ticks_labels` is `True`,
                                  tick labels will be shortened by taking the last `short_tick_labels_splits`
                                  parts of the layer name (split by '.'). For example, if a layer is
                                  "model.encoder.layer.0.attention" and `short_tick_labels_splits` is 2,
                                  the label will become "0-attention". If `None`, full layer names are used.
        use_tight_layout: A boolean indicating whether to automatically adjust plot parameters for a tight layout.
                          This helps prevent labels from overlapping or being cut off. Defaults to `True`.
        show_annotations: A boolean indicating whether to display the CKA values as text annotations on the heatmap cells.
                          Defaults to `True`.
        show_img: A boolean indicating whether to display the plot immediately after generation.
                  If `False`, the plot is generated but not shown (useful when only saving).
                  Defaults to `True`.
        show_half_heatmap: A boolean indicating whether to mask the upper triangle of the heatmap.
                           This is useful when the CKA matrix is symmetric and you only need to show
                           one half to avoid redundancy. Defaults to `False`.
        invert_y_axis: A boolean indicating whether to invert the y-axis of the plot.
                       Defaults to `True`.
        title_font_size: The font size for the plot title. Defaults to 14.
        axis_font_size: The font size for the x and y-axis labels. Defaults to 12.
        tick_font_size: The font size for the tick labels (layer names). Defaults to 10.
        figsize: A tuple `(width, height)` in inches, specifying the size of the figure.
                 Defaults to (10, 10).
        dpi: The dots per inch (resolution) for the saved figure. Higher values result in higher quality images.
             Defaults to 300.
    """
    # Build the mask
    mask = (
        torch.tril(torch.ones_like(cka_matrix, dtype=torch.bool), diagonal=-1)
        if show_half_heatmap
        else None
    )

    # Build the heatmap
    fig, ax = plt.subplots(figsize=figsize)
    ax = sn.heatmap(
        cka_matrix.cpu().numpy(),
        vmin=vmin,
        vmax=vmax,
        annot=show_annotations,
        cmap=cmap,
        mask=mask.cpu().numpy() if mask is not None else None,
        ax=ax,
    )
    if invert_y_axis:
        ax.invert_yaxis()

    ax.set_xlabel(f"{model2_name} Layers", fontsize=axis_font_size)
    ax.set_ylabel(f"{model1_name} Layers", fontsize=axis_font_size)

    # Deal with tick labels
    ax.set_xticks(np.arange(len(model2_layers)) + 0.5)
    ax.set_yticks(np.arange(len(model1_layers)) + 0.5)
    if show_ticks_labels:
        if short_tick_labels_splits is None:
            ax.set_xticklabels(
                model2_layers,
                fontsize=tick_font_size,
            )
            ax.set_yticklabels(
                model1_layers,
                fontsize=tick_font_size,
            )
        else:
            ax.set_xticklabels(
                [
                    "-".join(module.split(".")[-short_tick_labels_splits:])
                    for module in model2_layers
                ],
                fontsize=tick_font_size,
            )
            ax.set_yticklabels(
                [
                    "-".join(module.split(".")[-short_tick_labels_splits:])
                    for module in model1_layers
                ],
                fontsize=tick_font_size,
            )

        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Put the title if passed
    if title is not None:
        ax.set_title(title, fontsize=title_font_size)
    else:
        title = f"{model1_name} vs {model2_name}"
        ax.set_title(title, fontsize=title_font_size)

    # Set the layout to tight if the corresponding parameter is True
    if use_tight_layout:
        plt.tight_layout()

    # Save the plot to the specified path if defined
    if filepath is not None:
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
    elif dirpath is not None:
        title = title.replace("/", "-")
        path_rel = f"{dirpath}/{title}.png"
        plt.savefig(path_rel, dpi=dpi, bbox_inches="tight")

    # Show the image if the user chooses to do so
    if show_img:
        plt.show()
