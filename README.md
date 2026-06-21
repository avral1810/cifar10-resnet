# CIFAR-10 ResNet Vision Foundations

Learning-focused PyTorch project for CIFAR-10 image classification. The goal was not to copy a high-accuracy recipe, but to build the vision stack step by step: data loading, augmentation, CNN baselines, residual blocks, training loops, checkpointing, TensorBoard logging, and scheduler-driven experiments.

## Results

| experiment | model | augmentation | scheduler | test accuracy |
|---|---|---:|---:|---:|
| `baseline_cnn` | SimpleCNN | none | none | 65.57% |
| `baseline_cnn_aug_basic` | SimpleCNN | random crop + flip | none | 67.22% |
| `resnet_small_aug_basic` | ResNetSmall | random crop + flip | none | 83.09% |
| `resnet_small_aug_basic_cosine` | ResNetSmall | random crop + flip | cosine annealing | 84.72% |

Headline result: the project improved from a `65.57%` simple CNN baseline to `84.72%` with a from-scratch small ResNet, augmentation, and cosine learning-rate scheduling.

## What This Project Covers

- CIFAR-10 RGB tensors shaped `[batch, channels, height, width]`
- train/validation/test splitting with reproducible seeds
- train-set mean/std computation and cached normalization stats
- torchvision and Hugging Face dataset loading paths
- basic augmentation with random crop and horizontal flip
- simple CNN baseline
- residual block implementation with projection shortcuts
- small ResNet architecture for CIFAR-10
- train/eval loops with accuracy and loss metrics
- checkpointing the best validation model
- early stopping
- TensorBoard experiment logging
- GPU telemetry through `nvidia-smi`
- cosine learning-rate scheduling

## Project Structure

```text
cifar-10/
  configs/                 YAML experiment configs
  experiments/             result summaries and comparisons
  notes/                   learning notes from the build
  scripts/train.py         training entrypoint
  src/cifar10_resnet/
    dataset.py             CIFAR-10 loading, transforms, stats, loaders
    model.py               SimpleCNN, ResidualBlock, ResNetSmall
    train.py               one-epoch training loop
    evaluate.py            validation/test loop
    checkpoint.py          save/load best model checkpoints
    registry.py            model, optimizer, criterion, scheduler registries
    telemetry.py           GPU metric logging helpers
    utils.py               device selection and visualization helpers
```

Generated data, checkpoints, and TensorBoard runs are intentionally ignored by Git.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

This project was developed on WSL Ubuntu with an NVIDIA RTX 5080 and CUDA-enabled PyTorch.

Verify PyTorch sees the GPU:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

## Running Experiments

Run an experiment by passing a config name:

```bash
PYTHONPATH=src python scripts/train.py --config baseline_cnn.yaml
PYTHONPATH=src python scripts/train.py --config baseline_cnn_aug_basic.yaml
PYTHONPATH=src python scripts/train.py --config baseline_resnet_small.yaml
PYTHONPATH=src python scripts/train.py --config resnet_small_aug_basic_cosine.yaml
```

Launch TensorBoard:

```bash
tensorboard --logdir runs
```

## Main Takeaways

The simple CNN learned useful CIFAR-10 features, but it saturated around the mid-60s. Augmentation improved generalization slightly. The bigger jump came from residual connections: the small ResNet trained cleanly despite being much deeper than the baseline. Cosine annealing then improved the final result by letting the optimizer take larger early steps and smaller late steps.

See [experiments/README.md](experiments/README.md) for the experiment log and [notes/](notes/) for the learning notes.
