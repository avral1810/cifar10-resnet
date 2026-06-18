"""CIFAR-10 image classification experiments with CNN and ResNet models."""

from cifar10_resnet.dataset import (
    build_transform,
    compute_stats,
    create_dataloader,
    create_datasets,
    load_or_compute_stats,
    split_dataset,
)

__version__ = "0.1.0"

__all__ = [
    "build_transform",
    "compute_stats",
    "create_dataloader",
    "create_datasets",
    "load_or_compute_stats",
    "split_dataset",
]
