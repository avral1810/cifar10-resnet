import torch
import torch.nn as nn
from pathlib import Path


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    val_accuracy: float,
    optimizer: torch.optim.Optimizer,
    train_loss: float,
    val_loss: float, 
    path: str | Path="checkpoints"):
    try:
        if isinstance(path, str): 
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
        }, path)
        # print(f"Saved checkpoint at {path}")
    except Exception as e:
        print(f"Failed to save model: {e}")

def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    path: str | Path="checkpoints"
) -> dict[str]:
    if isinstance(path, str):
        path = Path(path)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    print(f"Loading checkpoint from {path} w/ val_loss: {val_loss} @ epoch: {epoch}\n")
    return {
        "model": model,
        'optimizer': optimizer,
        'epoch': epoch,
        'val_loss': val_loss,
        'train_loss':train_loss,
    }
