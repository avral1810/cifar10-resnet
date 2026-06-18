import math
from pathlib import Path

import torch
import matplotlib.pyplot as plt


def unnormalize_image(
    image: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    mean = mean.detach().cpu().view(3, 1, 1)
    std = std.detach().cpu().view(3, 1, 1)
    image = image.detach().cpu()
    image = image * std + mean
    image = image.clamp(0, 1)
    return image


def show_image(
    image: torch.Tensor,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    title: str | None = None,
) -> None:
    if mean is not None and std is not None:
        image = unnormalize_image(image, mean, std)
    image = image.detach().cpu()
    image = image.permute(1, 2, 0)
    plt.imshow(image)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show()

def show_feature_maps(
    features: torch.Tensor,
    max_maps: int = 8,
    title: str | None = None,
) -> None:
    features = features.detach().cpu()

    if features.ndim == 4:
        features = features[0]

    if features.ndim != 3:
        raise ValueError(f"Expected [C, H, W] or [1, C, H, W], got {features.shape}")

    num_maps = min(max_maps, features.shape[0])
    fig, axes = plt.subplots(1, num_maps, figsize=(num_maps * 2, 2))

    if num_maps == 1:
        axes = [axes]

    for i in range(num_maps):
        axes[i].imshow(features[i], cmap="gray")
        axes[i].set_title(f"ch {i}")
        axes[i].axis("off")

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()


def show_image_and_feature_maps(
    image: torch.Tensor,
    features: torch.Tensor,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    max_maps: int = 8,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> None:
    if mean is not None and std is not None:
        image = unnormalize_image(image, mean, std)
    else:
        image = image.detach().cpu().clamp(0, 1)

    image = image.permute(1, 2, 0)

    features = features.detach().cpu()
    if features.ndim == 4:
        features = features[0]

    if features.ndim != 3:
        raise ValueError(f"Expected features as [C, H, W] or [1, C, H, W], got {features.shape}")

    num_maps = min(max_maps, features.shape[0])
    num_plots = num_maps + 1
    num_cols = math.ceil(math.sqrt(num_plots))
    num_rows = math.ceil(num_plots / num_cols)

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 2, num_rows * 2),
    )
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    axes[0].imshow(image)
    axes[0].set_title("image")
    axes[0].axis("off")

    for i in range(num_maps):
        axes[i + 1].imshow(features[i], cmap="gray")
        axes[i + 1].set_title(f"ch {i}")
        axes[i + 1].axis("off")

    for axis in axes[num_plots:]:
        axis.axis("off")

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
