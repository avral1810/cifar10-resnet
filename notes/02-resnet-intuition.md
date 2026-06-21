# ResNet Intuition Notes

A normal CNN block learns:

```text
output = F(x)
```

A residual block learns:

```text
output = x + F(x)
```

The shortcut path carries the original signal forward. The convolution path learns a correction to that signal. This makes deeper CNNs easier to optimize because a block can behave close to identity if the extra layers are not useful yet.

## Shape Contract

Residual addition only works when both paths produce the same shape:

```text
main_path(x)     -> [B, C_out, H_out, W_out]
shortcut_path(x) -> [B, C_out, H_out, W_out]
```

If channels and spatial dimensions already match, the shortcut can be identity:

```text
[B, 16, 32, 32] + [B, 16, 32, 32]
```

If channels or spatial dimensions change, the shortcut needs a projection:

```text
Conv1x1(in_channels -> out_channels, stride=stride)
BatchNorm2d(out_channels)
```

Example:

```text
x:              [B, 16, 32, 32]
main path:      [B, 32, 16, 16]
shortcut path:  [B, 32, 16, 16]
sum:            [B, 32, 16, 16]
```

## Shape Rules

Convolution spatial output:

```text
out = floor((in + 2*padding - kernel_size) / stride) + 1
```

Useful rules:

- `out_channels` becomes the next layer's `in_channels`
- `stride=2` roughly halves height and width
- `padding=1` preserves spatial size for `kernel_size=3`, `stride=1`
- pooling changes height and width, not channel count
- `1x1` convolution is the standard way to mix or project channels

## Global Average Pooling

`AdaptiveAvgPool2d((1, 1))` averages each channel's spatial map into one value:

```text
[B, 64, 8, 8] -> [B, 64, 1, 1] -> flatten -> [B, 64]
```

This keeps the classifier small:

```text
Linear(64, 10)
```

instead of:

```text
Linear(64 * 8 * 8, 10)
```
