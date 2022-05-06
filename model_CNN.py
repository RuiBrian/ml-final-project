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
        self.n_conv_output = 640 if self.input_height == 82 else 3200

        # Convolutional Layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Dense Linear Layers
        self.linear_layers = nn.Sequential(
            nn.Linear(self.n_conv_output, 24), nn.ReLU(), nn.Linear(24, n_classes)
        )

    def forward(self, x):
        # Add channel and batch dimension
        x = x.unsqueeze(0)  # One channel
        x = x.unsqueeze(0)  # Batch size of 1

        # Pass through network layers
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)

        return x
