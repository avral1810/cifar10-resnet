from cifar10_resnet.model import SimpleCNN
import torch
import torch.nn as nn

from time import perf_counter

def train_one_epoch(
    model: SimpleCNN,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pin_memory: bool = False,
) -> float:
    model.train()
    train_loss = 0.0
    data_time, model_time = 0, 0
    for images, labels in dataloader:
        start = perf_counter()
        images = images.to(device, non_blocking=pin_memory)
        labels = labels.to(device, non_blocking=pin_memory)
        mid = perf_counter()
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        end = perf_counter()
        data_time += (mid - start)
        model_time += (end - mid)
    return train_loss / len(dataloader.dataset)
