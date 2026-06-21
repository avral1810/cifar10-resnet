"""CIFAR-10 image classification experiments with CNN and ResNet models."""

from cifar10_resnet.dataset import (
    build_transform,
    compute_stats,
    create_dataloader,
    create_datasets,
    load_or_compute_stats,
    split_dataset,
)
from cifar10_resnet.model import SimpleCNN, ResNetSmall
from cifar10_resnet.registry import (
    get_criterion_class,
    get_model_class,
    get_optimizer_class,
    get_scheduler_class,
)
from cifar10_resnet.telemetry import get_gpu_stats
from cifar10_resnet.train import train_one_epoch
from cifar10_resnet.evaluate import evaluate
from cifar10_resnet.utils import get_device, show_image_and_feature_maps, time_execution
from cifar10_resnet.checkpoint import save_checkpoint, load_checkpoint

__version__ = "0.1.0"

__all__ = [
    "build_transform",
    "compute_stats",
    "create_dataloader",
    "create_datasets",
    "load_or_compute_stats",
    "split_dataset",
    "SimpleCNN",
    "get_model_class",
    "get_optimizer_class",
    "get_criterion_class",
    "get_scheduler_class",
    "get_gpu_stats",
    "train_one_epoch",
    "show_image_and_feature_maps",
    "evaluate",
    "time_execution",
    "get_device",
    "load_checkpoint",
    "save_checkpoint",
    "ResNetSmall"
]
