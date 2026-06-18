import torch

from cifar10_resnet.dataset import create_dataloader, load_or_compute_stats
from cifar10_resnet.model import SimpleCNN
from cifar10_resnet.utils import show_image_and_feature_maps


def main():
    source = "hf"
    train_loader, _, _ = create_dataloader(batch_size=8, source=source)
    mean, std = load_or_compute_stats(source=source)

    images, labels = next(iter(train_loader))

    model = SimpleCNN()
    model.eval()

    with torch.no_grad():
        logits, features = model(images, return_features_after_nth_step=[3])

    print("image batch:", images.shape)
    print("label batch:", labels.shape)
    print("logits:", logits.shape)
    print("first label:", labels[0].item())
    print("feature layer:", features[0][0], features[0][1].shape)

    show_image_and_feature_maps(
        image=images[0],
        features=features[0][1][0],
        mean=mean,
        std=std,
        max_maps=8,
        title="Untrained conv1 feature maps",
        save_path="runs/untrained_conv1_features.png",
    )


if __name__ == "__main__":
    main()
