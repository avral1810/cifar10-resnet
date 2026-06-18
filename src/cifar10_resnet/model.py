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
