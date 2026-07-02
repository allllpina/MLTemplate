import torch
import torch.nn as nn
from efficient_kan import KAN


class HybridConvKAN(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        dropout_rate: float = 0.3,
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()

        self._to_linear = 32 * 2 * 2

        self.kan_classifier = KAN(
            [self._to_linear, 64, num_classes], grid_size=grid_size, spline_order=spline_order
        )

    def forward(self, x) -> torch.Tensor:  # type: ignore[no-untyped-def]
        x = self.encoder(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.flatten(x)

        logits = self.kan_classifier(x)
        return logits  # type: ignore[no-any-return]
