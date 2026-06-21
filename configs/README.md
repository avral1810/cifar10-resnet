# Configs

Experiment settings live in YAML files so model, data, optimizer, and scheduler choices can be changed without editing training code.

## Available Configs

| config | purpose |
|---|---|
| `baseline_cnn.yaml` | SimpleCNN baseline without augmentation |
| `baseline_cnn_aug_basic.yaml` | SimpleCNN with random crop and horizontal flip |
| `baseline_resnet_small.yaml` | ResNetSmall with basic augmentation |
| `resnet_small_aug_basic_cosine.yaml` | ResNetSmall with basic augmentation and cosine LR |

## Important Fields

`experiment.name` controls TensorBoard run names and checkpoint filenames.

`data.augmentation` controls train-time augmentation. When augmentation is enabled, preloading is disabled so random transforms are not frozen into one cached tensor.

`data.preload_to_device` is useful for tiny fixed datasets and no augmentation, but it is not a general strategy for larger vision work.

`model.name` is resolved through `src/cifar10_resnet/registry.py`.

`train.scheduler` is optional. The cosine config uses:

```yaml
scheduler:
  name: cosine_annealing
  T_max: 100
  eta_min: 0.00001
```

The scheduler steps once per epoch.
