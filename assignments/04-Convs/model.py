import torch
import torch.nn as nn


class Model(torch.nn.Module):
    """
    fast CNN
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=8, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2)
        self.fc = nn.Linear(16 * 6 * 6, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward
        """
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
