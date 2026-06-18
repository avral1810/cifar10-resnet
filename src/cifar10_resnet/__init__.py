"""CIFAR-10 image classification experiments with CNN and ResNet models."""

from cifar10_resnet.dataset import (
    build_transform,
    compute_stats,
    create_dataloader,
    create_datasets,
    load_or_compute_stats,
    split_dataset,
)
from cifar10_resnet.model import SimpleCNN
from cifar10_resnet.train import train_one_epoch
from cifar10_resnet.evaluate import evaluate
from cifar10_resnet.utils import get_device, show_image_and_feature_maps, time_execution

__version__ = "0.1.0"

__all__ = [
    "build_transform",
    "compute_stats",
    "create_dataloader",
    "create_datasets",
    "load_or_compute_stats",
    "split_dataset",
    "SimpleCNN",
    "train_one_epoch",
    "show_image_and_feature_maps",
    "evaluate",
    "time_execution",
    "get_device",
]
