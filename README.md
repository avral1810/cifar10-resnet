# CIFAR-10 ResNet Vision Foundations

Learning-focused PyTorch project for CIFAR-10 image classification.

Current scope:

- CIFAR-10 loading through Hugging Face or torchvision
- cached train-set mean/std computation
- simple CNN baseline
- feature-map visualization before training

Run the current smoke script:

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py
```

Generated data, checkpoints, and run artifacts are intentionally ignored by Git.
