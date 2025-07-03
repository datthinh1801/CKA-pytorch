import matplotlib.pyplot as plt
import seaborn as sn
import torch


def plot_cka(
    cka_matrix: torch.Tensor,
    model1_layers: list[str],
    model2_layers: list[str],
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
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
) -> None:
    """Plot the CKA matrix.

    Args:
        cka_matrix (torch.Tensor): The CKA matrix.
        model1_layers (list[str]): List of the names of the first model's layers.
        model2_layers (list[str]): List of the names of the second model's layers.
        model1_name (str): Name of the first model.
        model2_name (str): Name of the second model.
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
    # Build the mask
    mask = (
        torch.tril(torch.ones_like(cka_matrix, dtype=torch.bool), diagonal=-1)
        if show_half_heatmap
        else None
    )

    # Build the heatmap
    ax = sn.heatmap(
        cka_matrix.cpu().numpy(),
        vmin=vmin,
        vmax=vmax,
        annot=show_annotations,
        cmap=cmap,
        mask=mask.cpu().numpy() if mask is not None else None,
    )
    if invert_y_axis:
        ax.invert_yaxis()

    ax.set_xlabel(f"{model2_name} Layers", fontsize=12)
    ax.set_ylabel(f"{model1_name} Layers", fontsize=12)

    # Deal with tick labels
    if show_ticks_labels:
        if short_tick_labels_splits is None:
            ax.set_xticklabels(model2_layers)
            ax.set_yticklabels(model1_layers)
        else:
            ax.set_xticklabels(
                [
                    "-".join(module.split(".")[-short_tick_labels_splits:])
                    for module in model2_layers
                ]
            )
            ax.set_yticklabels(
                [
                    "-".join(module.split(".")[-short_tick_labels_splits:])
                    for module in model1_layers
                ]
            )

        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Put the title if passed
    if title is not None:
        ax.set_title(title, fontsize=14)
    else:
        title = f"{model1_name} vs {model2_name}"
        ax.set_title(title, fontsize=14)

    # Set the layout to tight if the corresponding parameter is True
    if use_tight_layout:
        plt.tight_layout()

    # Save the plot to the specified path if defined
    if save_path is not None:
        title = title.replace("/", "-")
        path_rel = f"{save_path}/{title}.png"
        plt.savefig(path_rel, dpi=400, bbox_inches="tight")

    # Show the image if the user chooses to do so
    if show_img:
        plt.show()
