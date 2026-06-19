from pathlib import Path
import warnings

from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
import torch
import json
import datasets

warnings.filterwarnings(
    "ignore",
    message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
    category=Warning,
    module=r"torchvision\.datasets\.cifar",
)

class HfCIFARxDataset(Dataset):
    
    def __init__(self, hf_dataset, transform=None) -> None:
        self.hf_dataset = hf_dataset
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index):
        sample = self.hf_dataset[index]
        image = sample["img"]
        label = sample["label"]

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label


def load_hf_cifar10(split: str):
    return datasets.load_dataset("uoft-cs/cifar10", split=split,  cache_dir="data/huggingface",)

def compute_stats(
    data_dir: str | Path="data/raw",
    batch_size: int = 128,
    source: str="torchvision",
) -> tuple[torch.Tensor, torch.Tensor]:
    transform = transforms.ToTensor()

    if source == "torchvision":
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            transform=transform,
            download=True,
        )
    elif source == "huggingface" or source == "hf":
        hf_train = load_hf_cifar10(split="train")
        train_dataset = HfCIFARxDataset(hf_dataset=hf_train, transform=transform)
    else:
        raise ValueError(f"Unknown source type: {source}")
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    num_pixels = 0

    for images, _ in train_loader:
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_squared_sum += (images ** 2).sum(dim=(0, 2, 3))
        num_pixels += images.shape[0] * images.shape[2] * images.shape[3]
    
    mean = channel_sum / num_pixels
    std = (channel_squared_sum / num_pixels - mean ** 2).sqrt()
    return mean, std


def load_or_compute_stats(
    data_dir: str | Path ="data/raw",
    batch_size : int=128,
    force_recompute: bool = False,
    source: str = "torchvision",
) -> tuple[torch.Tensor, torch.Tensor]:
    # check if data/processed has stats.json
    stats_path = Path(f"data/processed/cifar10_stats_{source}.json")
    if not stats_path.exists() or force_recompute:
        mean, std = compute_stats(
            data_dir=data_dir,
            batch_size=batch_size,
            source=source,
        )
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w") as op_file:
            json.dump(
                {
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                },
                op_file,
                indent=2
            )
        return mean, std
    with stats_path.open() as ip_file:
        stats = json.load(ip_file)
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)    
    return mean, std


def build_transform(
    mean: torch.Tensor,
    std: torch.Tensor,
    augmentation: str | None=None,
    train: bool=True,
) -> tuple[transforms.Compose, transforms.Compose]:
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean, std)
    if not train or augmentation is None:
        transform = transforms.Compose([
            to_tensor,
            normalize,
        ])
    elif augmentation == "basic":
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                to_tensor,
                normalize,
            ]
        )
    else:
        raise ValueError(f"Unknown augmentation `{augmentation}`")
    return transform


def preload_dataset(
    dataset: Dataset,
    device: torch.device,
    preload_to_device: bool=False,
):
    print(f"Preloading Dataset to {device.type if preload_to_device else 'cpu'} Memory")
    images = []
    labels = []

    for image, label in dataset:
        images.append(image)
        labels.append(label)
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    if preload_to_device:
        images = images.to(device)
        labels = labels.to(device)
    return TensorDataset(images, labels)


def create_datasets(
    device: torch.device,
    batch_size: int=128,
    source: str="torchvision",
    preload: bool=False,
    preload_to_device: bool=False,
    augmentation: str | None=None,
    seed: int=42,
    val_frac: float=0.1,
    data_dir: str | Path="data/raw",
) -> tuple[Dataset, Dataset, Dataset]:
    mean, std = load_or_compute_stats(data_dir=data_dir, batch_size=batch_size, source=source)
    train_transform, eval_tranform = build_transform(mean, std, augmentation=augmentation, train=True), build_transform(mean, std, augmentation=augmentation, train=False)
   
    if source == "torchvision":
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=train_transform
        )
        val_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=eval_tranform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=False,
            transform=eval_tranform,
            download=True
        )
    elif source in {"huggingface", "hf"}:
        hf_train = load_hf_cifar10("train")
        hf_test = load_hf_cifar10("test")
        train_dataset = HfCIFARxDataset(hf_train, train_transform)
        val_dataset = HfCIFARxDataset(hf_train, eval_tranform)
        test_dataset = HfCIFARxDataset(hf_test, eval_tranform)
    else:
        raise ValueError(f"Unknown source: {source}")
    train_indices, val_indices = split_dataset(
        len(train_dataset),
        val_frac=val_frac,
        seed=seed,
    )
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)
    if preload:
        train_dataset, val_dataset, test_dataset = preload_dataset(
            dataset=train_dataset, 
            device=device, 
            preload_to_device=preload_to_device
        ), preload_dataset(
            dataset=val_dataset,
            device=device, 
            preload_to_device=preload_to_device
        ), preload_dataset(
            dataset=test_dataset,
            device=device, 
            preload_to_device=preload_to_device
        )
    return train_dataset, val_dataset, test_dataset

def create_dataloader(
    device: torch.device,
    batch_size: int=128,
    num_workers:int=0,
    val_frac: float=0.1,
    seed: int=42,
    source: str="torchvision",
    pin_memory: bool=False,
    preload: bool=False,
    preload_to_device: bool=False,
    augmentation: str | None=None,
    data_dir: str | Path="data/raw",

) -> tuple[DataLoader, DataLoader, DataLoader]:
    train, val, test = create_datasets(
        device=device,
        data_dir=data_dir,
        batch_size=batch_size,
        source=source,
        preload=preload,
        preload_to_device=preload_to_device,
        augmentation=augmentation,
        seed=seed,
        val_frac=val_frac,
    )
    train_loader = DataLoader(
        dataset=train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        dataset=val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        dataset=test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader

def split_dataset(
    size: int,
    val_frac: float=0.1,
    seed: int=42,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(size, generator=generator)
    val_size = int(size * val_frac)
    return indices[val_size:], indices[:val_size]
