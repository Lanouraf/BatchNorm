import torch.nn as nn

from BatchNorm import BatchNorm2D, BatchNorm1D


class LeNet(nn.Module):
    """
    LeNet-5 architecture
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size = 5, padding = 2),  # C1: 28x28x6
            nn.AvgPool2d(kernel_size = 2, stride = 2),  # S2: 14x14x6
            nn.Sigmoid(),

            nn.Conv2d(6, 16, kernel_size = 5),  # C3: 10x10x16
            nn.AvgPool2d(kernel_size = 2, stride = 2),  # 5x5x16
            nn.Sigmoid(),

            nn.Flatten(),
            nn.Linear(5 * 5 * 16, 120),
            nn.Sigmoid(),

            nn.Linear(120, 84),
            nn.Sigmoid(),

            nn.Linear(84, 10),
            nn.Softmax()
        )

    def forward(self, x):
        return self.layers(x)


class LeNetBN(nn.Module):
    """
    LeNet-5 architecture with BatchNorm blocks
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size = 5, padding = 2),  # C1: 28x28x6
            nn.AvgPool2d(kernel_size = 2, stride = 2),  # S2: 14x14x6
            BatchNorm2D(6),
            nn.Sigmoid(),

            nn.Conv2d(6, 16, kernel_size = 5),  # C3: 10x10x16
            nn.AvgPool2d(kernel_size = 2, stride = 2),  # 5x5x16
            BatchNorm2D(16),
            nn.Sigmoid(),

            nn.Flatten(),
            nn.Linear(5 * 5 * 16, 120),
            BatchNorm1D(120),
            nn.Sigmoid(),

            nn.Linear(120, 84),
            BatchNorm1D(84),
            nn.Sigmoid(),

            nn.Linear(84, 10),
            nn.Softmax()
        )

    def forward(self, x):
        return self.layers(x)


class LeNetStockBN(nn.Module):
    """
    LeNet-5 architecture with Pytorch BatchNorm blocks
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size = 5, padding = 2),  # C1: 28x28x6
            nn.AvgPool2d(kernel_size = 2, stride = 2),  # S2: 14x14x6
            nn.BatchNorm2d(6),
            nn.Sigmoid(),

            nn.Conv2d(6, 16, kernel_size = 5),  # C3: 10x10x16
            nn.AvgPool2d(kernel_size = 2, stride = 2),  # 5x5x16
            nn.BatchNorm2d(16),
            nn.Sigmoid(),

            nn.Flatten(),
            nn.Linear(5 * 5 * 16, 120),
            nn.BatchNorm1d(120),
            nn.Sigmoid(),

            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.Sigmoid(),

            nn.Linear(84, 10),
            nn.Softmax()
        )

    def forward(self, x):
        return self.layers(x)
