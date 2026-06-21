import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn_layers = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=16, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        ])

        self.flatten = nn.Flatten()

        self.ff = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x, return_features_after_nth_step=None):
        return_features = []
        if return_features_after_nth_step is None:
            return_features_after_nth_step = set()
        else:
            return_features_after_nth_step = set(return_features_after_nth_step)
        for i, layer  in enumerate(self.cnn_layers):
            x = layer(x)
            if return_features_after_nth_step is not None and i in return_features_after_nth_step:
                return_features.append((i, x))
        if return_features_after_nth_step is not None and len(return_features_after_nth_step) > 0:
            return self.ff(self.flatten(x)), return_features 
        return self.ff(self.flatten(x))

class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=16, stride=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=stride
            ),
            # out = (in + 2*padding - kernel_size) / stride) + 1
            # out => ((16 + 2 - 3) / 2) + 1 => 8
 
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            # => 8
            nn.BatchNorm2d(num_features=out_channels),
        )
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        self.final_activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor):
        return self.final_activation(self.shortcut(x) + self.cnn(x))

class ResNetSmall(nn.Module):
    def __init__(self, in_channels=3, stride=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=16,
                stride=stride,
                padding=1,
                kernel_size=3,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock(16, 16, stride=1),
            ResidualBlock(16, 16, stride=1),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(16, 32, stride=2),
            ResidualBlock(32, 32, stride=1),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64, stride=1),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10)
        )

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        for stage in (self.stage1, self.stage2, self.stage3):
            x = stage(x)
        x = self.classifier(x)
        return x
