from pathlib import Path
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torch
import json
import datasets

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


def build_transform(mean: torch.Tensor, std: torch.Tensor) -> transforms.Compose:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )
    return transform

def create_datasets(
    data_dir: str | Path="data/raw",
    batch_size: int=128,
    source: str="torchvision",
) -> tuple[Dataset, Dataset]:
    mean, std = load_or_compute_stats(data_dir=data_dir, batch_size=batch_size, source=source)
    transform = build_transform(mean, std)

    if source == "torchvision":
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=False,
            transform=transform,
            download=True
        )
    elif source in {"huggingface", "hf"}:
        hf_train = load_hf_cifar10("train")
        hf_test = load_hf_cifar10("test")
        train_dataset = HfCIFARxDataset(hf_train, transform)
        test_dataset = HfCIFARxDataset(hf_test, transform)

    else:
        raise ValueError(f"Unknown source: {source}")
    return train_dataset, test_dataset

def create_dataloader(
    data_dir: str | Path="data/raw",
    batch_size: int=128,
    num_workers:int=0,
    val_frac: float=0.1,
    seed: int=42,
    source: str="torchvision",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train, test = create_datasets(
        data_dir=data_dir,
        batch_size=batch_size,
        source=source,
    )
    train, val = split_dataset(train, val_frac=val_frac, seed=seed)
    train_loader = DataLoader(
        dataset=train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        dataset=val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        dataset=test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader

def split_dataset(
    dataset: torchvision.datasets.CIFAR10,
    val_frac: float=0.1,
    seed: int=42,
) -> tuple[Subset, Subset]:
    val_size = int(len(dataset) * val_frac)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train, val = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )
    return train, val