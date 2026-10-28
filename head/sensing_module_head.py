"""
Copyright (c) 2026 Lei Guo
email: l.guo3@uq.edu.au
Institution: The University of Queensland

This file is part of the "Physics-Guided Coupled Neural Networks for
Clutter Extraction in Medical Microwave Sensing" project.
Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""


import torch
import torch.nn as nn


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class SensingModule(nn.Module):
    def __init__(self, channel_num):
        super(SensingModule, self).__init__()

        self.four_conv2d = nn.Sequential(
            nn.Conv2d(2, channel_num[0], kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(channel_num[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(channel_num[0], channel_num[1], kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(channel_num[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(channel_num[1], channel_num[2], kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(channel_num[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(channel_num[2], channel_num[3], kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(channel_num[3]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(channel_num[3], channel_num[4], kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(channel_num[4]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        self.linear_layer = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):

        x = self.four_conv2d(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer(x.squeeze())

        return x
