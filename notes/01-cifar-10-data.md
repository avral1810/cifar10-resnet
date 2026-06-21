# CIFAR-10 Data Notes

CIFAR-10 images are small RGB images:

```text
single image before ToTensor: [height, width, channels] = [32, 32, 3]
single image after ToTensor:  [channels, height, width] = [3, 32, 32]
batch:                        [B, 3, 32, 32]
```

The three input channels are red, green, and blue. After the first convolution, channels are no longer RGB. They become learned feature channels.

## Normalization

Normalization uses mean and standard deviation computed over the training set. For CIFAR-10 this is computed per RGB channel, so the result has three means and three standard deviations.

The reduction dimensions are:

```text
images: [B, C, H, W]
mean over B, H, W
keep C
```

That is why the stats computation reduces over dimensions `0`, `2`, and `3`.

Normalization helps the optimizer because inputs arrive at the model on a more stable scale.

## Augmentation

The basic augmentation experiment uses:

- random crop with padding
- random horizontal flip
- normalization

Only the training split should receive random augmentation. Validation and test should use deterministic transforms so metrics are stable.

## Splitting Train and Validation

When train and validation need different transforms, they should not be made from the same transformed dataset object. The project uses shared indices over separate dataset instances:

```text
train dataset: CIFAR train split + train transforms
val dataset:   CIFAR train split + eval transforms
test dataset:  CIFAR test split + eval transforms
```

This keeps augmentation out of validation while preserving a reproducible split.
