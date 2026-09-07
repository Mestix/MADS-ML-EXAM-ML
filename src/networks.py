import torch
from torch import nn

# simpele 1D CNN
class CNN1D(nn.Module):
    def __init__(
        self,
        features: int = 1,
        num_classes: int = 5,
        kernel_size: int = 5,
        filters: int = 16,
        input_size: int = 192,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.filters = filters
        self.input_size = input_size

        self.convolutions = nn.Sequential(
            nn.Conv1d(
                in_channels=features,
                out_channels=filters,
                kernel_size=kernel_size,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # 192 meetpunten worden na pooling 96 meetpunten (de helft)
        flatten_size = filters * (input_size // 2)

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dataset geeft is vorm (batch, 192, 1)
        # Conv1d verwacht (batch, 1, 192) dus omdraaien
        
        x = x.permute(0, 2, 1)

        x = self.convolutions(x)
        logits = self.dense(x)

        return logits

# Paralelle 2D CNN
class ParallelCNN2D(nn.Module):
    def __init__(
        self,
        num_classes=5,
        filters=16,
        matrixshape=(16, 12),
        large_kernel=5,
        dropout=0.0,
    ):
        super().__init__()

        self.small_route = nn.Sequential(
            nn.Conv2d(
                1,
                filters,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        large_padding = large_kernel // 2

        self.large_route = nn.Sequential(
            nn.Conv2d(
                1,
                filters,
                kernel_size=large_kernel,
                padding=large_padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        pooled_height = matrixshape[0] // 2
        pooled_width = matrixshape[1] // 2

        flatten_size = (
            2
            * filters
            * pooled_height
            * pooled_width
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(
                flatten_size,
                num_classes,
            ),
        )

    def forward(self, x):
        small = self.small_route(x)
        large = self.large_route(x)

        x = torch.cat(
            [small, large],
            dim=1,
        )

        return self.classifier(x)
    
class CNN2D(nn.Module):
    def __init__(
        self,
        features: int = 1,
        num_classes: int = 5,
        kernel_size: int = 3,
        filters: int = 16,
        matrixshape: tuple = (16, 12),
    ) -> None:
        super().__init__()

        self.convolutions = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=filters,
                kernel_size=kernel_size,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        flatten_size = (
            filters
            * (matrixshape[0] // 2)
            * (matrixshape[1] // 2)
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutions(x)
        logits = self.dense(x)
        return logits