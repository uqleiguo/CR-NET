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
import torch.optim as optim


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class AE_Module(nn.Module):
    def __init__(self, ch_in, depth):
        super(AE_Module, self).__init__()

        self.depth = depth

        self.cnn_encoder = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(in_channels=ch_in * 2 ** n, out_channels=ch_in * 2 ** (n + 1), kernel_size=3, stride=1,
                          padding=1),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
                nn.BatchNorm1d(ch_in * 2 ** (n + 1)),
                nn.ReLU()) for n in range(depth)])

        self.cnn_decoder = nn.ModuleList(
            [nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv1d(in_channels=ch_in * 2 ** n, out_channels=ch_in * 2 ** (n - 1), kernel_size=3,
                            stride=1, padding=1),
                nn.BatchNorm1d(ch_in * 2 ** (n - 1)),
                nn.ReLU()) for n in range(depth, 0, -1)])

    def forward(self, x):
        x_skip = []
        # CR_Modual encoder
        for n, layer in enumerate(self.cnn_encoder):
            x = layer(x)
            x_skip.append(x)

        x_skip.pop()
        # CR_Modual decoder
        for n, layer in enumerate(self.cnn_decoder):
            x = layer(x)
            if n is not self.depth - 1:
                x = x + x_skip.pop()

        return x


def train(model, train_set, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for ii, data in enumerate(train_set):
        src = data[0].to(device, torch.float)
        src = torch.abs(src[:, :, 8:-9])        # Reduce the length of the input signal to 896 so the AE can equalize the output signal length from encoder and decoder

        optimizer.zero_grad()

        x = model(src)

        loss = criterion(x, src)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_set)


if __name__ == '__main__':
    train_dataset = torch.load('../data/train_dataset_skin.pt', weights_only=False)

    CLIP = 1
    learning_rate = 2e-5
    model = AE_Module(ch_in=2, depth=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    epoch_num = 25000
    train_loss = torch.zeros(epoch_num)

    min_loss = 1e9
    for epoch in range(0, epoch_num):
        train_loss[epoch] = train(model, train_dataset, optimizer, criterion, CLIP)

        print(f"Train Loss at the {epoch}th epoch is {train_loss[epoch]}\n")
