import torch
import torch.nn as nn
from cifar10_resnet.model import SimpleCNN


def evaluate(
        model: SimpleCNN,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: torch.device,
        pin_memory: bool = False,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)
            logits = model(images)
            loss = criterion(logits, labels)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * images.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy
