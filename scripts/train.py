import torch
import torch.nn as nn
from time import perf_counter
import yaml

from cifar10_resnet import (
    create_dataloader,
    evaluate,
    get_criterion_class,
    get_device,
    get_model_class,
    get_optimizer_class,
    show_image_and_feature_maps,
    time_execution,
    train_one_epoch,
)

def get_config(file_name: str="configs/baseline_cnn.yaml"):
    with open(file_name, "r") as ip_file:
        config = yaml.safe_load(ip_file)
    return config

@time_execution
def get_loaders(
    source: str,
    device: torch.Device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool=False,
    preload: bool=False,
    preload_to_device: bool=False,
    seed: int=42,
):
    if preload and preload_to_device and (num_workers != 0 or pin_memory is not False):
        print("Overriding `num_workers` to 0 becuase of preloading to GPU")
        num_workers = 0
        pin_memory = False
        
    train_loader, val_loader, test_loader = create_dataloader(
        device=device,
        source=source,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        preload=preload,
        preload_to_device=preload_to_device,
        seed=seed,
    )
    return train_loader, val_loader, test_loader

@time_execution
def main():
    config = get_config()
    device_str = config["runtime"].get("device", "auto")
    
    source=config["data"].get("source", "torchvision")
    batch_size = config["data"].get("batch_size", 1024)
    num_workers = config["data"].get("num_workers", 0)
    pin_memory = config["data"].get("pin_memory", False) 
    preload = config["data"].get("preload", False)
    preload_to_device = config["data"].get("preload_to_device", False)
    seed = config["experiment"].get("seed", 42)
    exp_name = config["experiment"].get("name", "Unknown")

    print(f"===Running Experiment `{exp_name}`===")
     
    if preload and preload_to_device and (num_workers != 0 or pin_memory is not False):
        print("Overriding `num_workers` to 0 becuase of preloading to GPU")
        num_workers = 0
        pin_memory = False
    
    if device_str == "auto":
        device = get_device()
    else:
        device = torch.device(device_str)
    train_loader, val_loader, test_loader = get_loaders(
        source=source,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        preload=preload, 
        preload_to_device=preload_to_device,
        seed=seed,
    )

    model = get_model_class(config["model"]["name"])()
    criterion = get_criterion_class(config["train"]["criterion"])()
    optimizer = get_optimizer_class(config["train"]["optimizer"])(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"].get("weight_decay", 0.0),
    )
    model = model.to(device)
    epochs = config["train"]["epochs"]

    for epoch in range(epochs):
        start = perf_counter()
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            pin_memory=pin_memory,
        )
        val_loss, val_accuracy = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            pin_memory=pin_memory,

        )
        end = perf_counter() - start
        if epoch < 5 or epoch % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_accuracy={val_accuracy:.2%} | "
                f"completed in {end:.2f} seconds"
            )
    
    test_loss, test_accuracy = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        pin_memory=config["data"].get("pin_memory", False),
    )
    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"test_loss={test_loss:.4f} | "
        f"test_accuracy={test_accuracy:.2%} | "
    )



    # show_image_and_feature_maps(
    #     image=images[0],
    #     features=features[0][1][0],
    #     mean=mean,
    #     std=std,
    #     max_maps=8,
    #     title="Untrained conv1 feature maps",
    #     save_path="runs/untrained_conv1_features.png",
    # )


if __name__ == "__main__":
    main()
