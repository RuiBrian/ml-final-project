import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class CNN(torch.nn.Module):
    def __init__(self, input_height, input_width, n_classes):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.n_classes = n_classes

        # ConvNet architecture
        self.conv_layers = nn.Sequential(
            # TODO: Tweak model architecture
            nn.Conv2d(1, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=1, stride=1),
            nn.Conv2d(128, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=1, stride=1),
            nn.Conv2d(128, 64, kernel_size=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=1, stride=1),
            nn.Conv2d(64, 64, kernel_size=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=1, stride=1),
            nn.Conv2d(64, 8, kernel_size=1, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=1, stride=1),
        )

        self.linear_layers = nn.Sequential(nn.Linear(48, self.n_classes))

    def forward(self, x):
        # Add channel and batch dimension
        x = x.unsqueeze(0)  # One channel
        x = x.unsqueeze(0)  # Batch size of 1

        # Pass through network layers
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)

        return x
