# Experiment Log

This file tracks the controlled experiment path for the CIFAR-10 vision foundations project. Each row changes one major idea at a time so the result is interpretable.

## Summary

| experiment | config | model | augmentation | scheduler | best validation accuracy | test accuracy |
|---|---|---|---:|---:|---:|---:|
| baseline CNN | `baseline_cnn.yaml` | SimpleCNN | none | none | ~66.00% | 65.57% |
| baseline CNN + augmentation | `baseline_cnn_aug_basic.yaml` | SimpleCNN | random crop + flip | none | 67.84% | 67.22% |
| small ResNet + augmentation | `baseline_resnet_small.yaml` | ResNetSmall | random crop + flip | none | 82.46% | 83.09% |
| small ResNet + augmentation + cosine LR | `resnet_small_aug_basic_cosine.yaml` | ResNetSmall | random crop + flip | cosine annealing | not recorded here | 84.72% |

## Experiment Notes

### 1. Simple CNN Baseline

The first model used two convolution blocks followed by a small fully connected classifier. It established that the data pipeline, normalization, training loop, evaluation loop, and checkpointing worked.

Result: `65.57%` test accuracy.

### 2. Simple CNN With Augmentation

Basic augmentation used random crop with padding and random horizontal flip. Training became much slower because transforms were applied during training, but generalization improved.

Result: `67.22%` test accuracy.

Takeaway: augmentation helped, but the small CNN did not have enough capacity to make a large jump.

### 3. ResNetSmall With Augmentation

The ResNet model added residual blocks with projection shortcuts for channel and spatial changes. This was the largest improvement in the project.

Result: `83.09%` test accuracy.

Takeaway: residual connections made a deeper CNN practical to train and improved CIFAR-10 accuracy substantially.

### 4. ResNetSmall With Cosine LR

Cosine annealing decayed the learning rate from `0.001` toward `0.00001` over the run. This gave the model a stronger late-training finish.

Result: `84.72%` test accuracy.

Takeaway: scheduler tuning gave a smaller but meaningful improvement after the architecture was already strong.

## Reproduction Commands

```bash
PYTHONPATH=src python scripts/train.py --config baseline_cnn.yaml
PYTHONPATH=src python scripts/train.py --config baseline_cnn_aug_basic.yaml
PYTHONPATH=src python scripts/train.py --config baseline_resnet_small.yaml
PYTHONPATH=src python scripts/train.py --config resnet_small_aug_basic_cosine.yaml
```

TensorBoard:

```bash
tensorboard --logdir runs
```
