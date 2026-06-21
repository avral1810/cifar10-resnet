import torch
from time import perf_counter
from datetime import datetime
from pathlib import Path
import yaml
from torch.utils.tensorboard import SummaryWriter

from argparse import ArgumentParser

from cifar10_resnet import (
    create_dataloader,
    evaluate,
    get_criterion_class,
    get_device,
    get_gpu_stats,
    get_model_class,
    get_optimizer_class,
    get_scheduler_class,
    show_image_and_feature_maps,
    time_execution,
    train_one_epoch,
    save_checkpoint,
    load_checkpoint,
    ResNetSmall,
)

def get_config(file_name: str="configs/baseline_cnn.yaml"):
    with open(file_name, "r") as ip_file:
        config = yaml.safe_load(ip_file)
    return config


def create_writer(config: dict) -> SummaryWriter:
    run_dir = Path(config["outputs"].get("run_dir", "runs"))
    experiment_name = config["experiment"].get("name", "experiment")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return SummaryWriter(run_dir / f"{experiment_name}-{timestamp}")

@time_execution
def get_loaders(
    source: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    val_frac: float,
    pin_memory: bool=False,
    preload: bool=False,
    preload_to_device: bool=False,
    seed: int=42,
    augmentation: str | None=None,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if preload and preload_to_device and (num_workers != 0 or pin_memory is not False):
        print("Overriding `num_workers` to 0 because of preloading to GPU")
        num_workers = 0
        pin_memory = False
        
    train_loader, val_loader, test_loader = create_dataloader(
        device=device,
        source=source,
        batch_size=batch_size,
        num_workers=num_workers,
        val_frac=val_frac,
        pin_memory=pin_memory,
        preload=preload,
        preload_to_device=preload_to_device,
        seed=seed,
        augmentation=augmentation,
    )
    # print(
    #     f"Train DataLoader: {train_loader.dataset.dataset.transform} | {len(train_loader.dataset.dataset)}\n",
    #     f"Val DataLoader: {val_loader.dataset.dataset.transform} | {len((val_loader.dataset.dataset))}\n",
    #     f"Test DataLoader: {test_loader.dataset.transform} | {len((test_loader.dataset))}\n",
    # )
    return train_loader, val_loader, test_loader

@time_execution
def main(config_file_name: str="configs/baseline_cnn.yaml"):

    config = get_config(file_name=config_file_name)
    device_str = config["runtime"].get("device", "auto")
    
    source=config["data"].get("source", "torchvision")
    batch_size = config["data"].get("batch_size", 1024)
    num_workers = config["data"].get("num_workers", 0)
    val_frac = config["data"].get("val_frac", 0.1)
    pin_memory = config["data"].get("pin_memory", False) 
    preload = config["data"].get("preload", False)
    preload_to_device = config["data"].get("preload_to_device", False)
    augmentation = config["data"].get("augmentation", "basic")
    seed = config["experiment"].get("seed", 42)
    exp_name = config["experiment"].get("name", "Unknown")

    if augmentation is not None:
        print(f"Overriding `preload_to_device`, `preload` to False for augmentation {augmentation}\n")
        preload_to_device = preload = False

    print(f"===Running Experiment `{exp_name}`===")
     
    if preload and preload_to_device and (num_workers != 0 or pin_memory is not False):
        print("Overriding `num_workers` to 0 because of preloading to GPU")
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
        val_frac=val_frac,
        pin_memory=pin_memory, 
        preload=preload, 
        preload_to_device=preload_to_device,
        seed=seed,
        augmentation=augmentation,
    )
    model = get_model_class(config["model"]["name"])()
    criterion = get_criterion_class(config["train"]["criterion"])()
    optimizer = get_optimizer_class(config["train"]["optimizer"])(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"].get("weight_decay", 0.0),
    )
    scheduler = None
    scheduler_config = config["train"].get("scheduler")
    if scheduler_config is not None:
        scheduler_class = get_scheduler_class(scheduler_config["name"])
        scheduler_kwargs = {
            key: value
            for key, value in scheduler_config.items()
            if key != "name"
        }
        scheduler = scheduler_class(optimizer, **scheduler_kwargs)

    model = model.to(device)
    epochs = config["train"]["epochs"]
    writer = create_writer(config)
    writer.add_text("config/yaml", f"```yaml\n{yaml.safe_dump(config)}\n```")
    best_val_accuracy = -1.0
    strike = 0
    checkpoint_path = f'{config["outputs"].get("checkpoint_dir", "checkpoints")}/{exp_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.pt' 
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
        step = epoch + 1
        writer.add_scalar("loss/train", train_loss, step)
        writer.add_scalar("loss/val", val_loss, step)
        writer.add_scalar("accuracy/val", val_accuracy, step)
        writer.add_scalar("time/epoch_seconds", end, step)
        writer.add_scalar("optimizer/learning_rate", optimizer.param_groups[0]["lr"], step)

        for name, value in get_gpu_stats(device).items():
            writer.add_scalar(name, value, step)
        
        if val_accuracy > best_val_accuracy:
            # print(f"Saving new Model checkpoint with accuracy: {val_accuracy:.2%}, epoch: {epoch}\n")
            best_val_accuracy = val_accuracy
            save_checkpoint(
                model,
                train_loss=train_loss,
                val_loss=val_loss,
                optimizer=optimizer,
                epoch=epoch,
                val_accuracy=val_accuracy,
                path=checkpoint_path,
            )
            # reset strikes in case of 2 bad then a good one,
            # n consicutive bad epochs stops the run
            strike = 0
        else:
            strike += 1

        if epoch < 5 or epoch % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_accuracy={val_accuracy:.2%} | "
                f"completed in {end:.2f} seconds"
            )
        if strike >= config["train"].get('early_stop_strike', 5):
            print(f"Early Stop oppurtunity, stopping iterations at {val_accuracy:.2%}")
            break
        if scheduler is not None:
            scheduler.step()

    checkpoint = load_checkpoint(
        model=model,
        optimizer=optimizer,
        device=device,
        path=checkpoint_path,
    )
    test_loss, test_accuracy = evaluate(
        model=checkpoint['model'],
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        pin_memory=pin_memory,
    )
    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"test_loss={test_loss:.4f} | "
        f"test_accuracy={test_accuracy:.2%} | "
    )
    writer.add_scalar("loss/test", test_loss, epoch + 1)
    writer.add_scalar("accuracy/test", test_accuracy, epoch + 1)
    writer.close()



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
    # main2()
    parser = ArgumentParser()
    parser.add_argument("--config", "-c", type=str, help="config file", default="baseline_cnn.yaml")
    args = parser.parse_args()
    config_file_name = f"configs/{args.config}"
    main(config_file_name=config_file_name)
