# Training And Debugging Notes

## Reproducibility

Seeds are useful for:

- train/validation splits
- model initialization
- dataloader shuffling
- augmentation randomness

They do not guarantee bit-for-bit identical results on every GPU setup, but they make experiments much easier to compare.

## Checkpointing

The training script saves a checkpoint whenever validation accuracy improves. Early stopping may stop later than the best epoch. For example:

```text
epoch 63: best validation accuracy, checkpoint saved
epoch 73: early stopping triggers after 10 weaker epochs
```

The final test should load the best checkpoint, not necessarily the last epoch's weights.

## Early Stopping

Small baselines can use lower patience values like `3` or `5`. Deeper augmented models often need more patience because validation accuracy can wobble before improving again.

For ResNetSmall with augmentation, `10` to `15` is a reasonable patience range.

## Data Loading Performance

For tiny fixed datasets, preloading tensors can make epochs very fast. For augmented training, preloading transformed tensors is usually wrong because it freezes random transforms.

General rules:

- `pin_memory=True` helps CPU-to-CUDA transfer when batches are copied to GPU
- `non_blocking=True` pairs with pinned memory
- `num_workers=0` means the main process loads data; it does not mean auto
- too many workers can slow small datasets down
- preloading to GPU is useful for CIFAR-scale experiments, not large vision datasets

## TensorBoard

The project logs:

- train loss
- validation loss
- validation accuracy
- test loss
- test accuracy
- epoch time
- learning rate
- GPU utilization and memory metrics

TensorBoard reads event files from `runs/`; it does not collect metrics automatically unless the training script writes them.

## Learning Rate Scheduling

Cosine annealing starts from the optimizer learning rate and gradually decays toward a minimum:

```text
0.001 -> 0.00001
```

In this project the scheduler steps once per epoch. It improved the ResNetSmall result from `83.09%` to `84.72%` test accuracy.
