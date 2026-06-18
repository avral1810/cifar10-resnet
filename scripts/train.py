import torch
import torch.nn as nn
from time import perf_counter

from cifar10_resnet import (
    SimpleCNN,
    create_dataloader,
    show_image_and_feature_maps,
    train_one_epoch,
    evaluate,
    get_device,
    time_execution,
)

@time_execution
def main():
    source = "hf"
    device = get_device()
    print(f"Using Device: {device}")
    batch_size = 1024
    pin_memory = device.type == "cuda"
    train_loader, val_loader, test_loader = create_dataloader(
        source=source,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=pin_memory,
    )
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model = model.to(device)
    epochs = 10

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
        pin_memory=pin_memory,
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
