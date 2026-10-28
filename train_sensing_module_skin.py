"""
Copyright (c) 2026 Lei Guo
email: l.guo3@uq.edu.au
Institution: The University of Queensland

This file is part of the "Physics-Guided Coupled Neural Networks for
Clutter Extraction in Medical Microwave Sensing" project.
Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from train_clutter_extractor_skin import AE_Module
from sensing_module_skin import AlexNet


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def signal_normalization(x, epsilon=1e-8):
    x_min = x.amin(dim=1, keepdim=True)   # shape [B, 1]
    x_max = x.amax(dim=1, keepdim=True)   # shape [B, 1]
    range_ = x_max - x_min      # shape [B, 1]

    x_norm = (x - x_min) / (range_ + epsilon)

    return x_norm


class CR_NET(nn.Module):
    def __init__(self, AE_in, AE_depth):
        super(CR_NET, self).__init__()

        self.clutter_gen = AE_Module(ch_in=AE_in, depth=AE_depth)
        self.channel_nums = [16, 32, 64, 128, 256]
        self.alexnet = AlexNet(self.channel_nums)

    def forward(self, x):
        x_input = torch.abs(x)

        x = self.clutter_gen(x)

        x_tr_raw = torch.abs(x_input - x)
        bs, c, k = x_tr_raw.shape
        x_tr_new = x_tr_raw.reshape(bs * c, k)
        x_tr = signal_normalization(x_tr_new).view(bs, c, k).to(device)

        x_tv = torch.unsqueeze(torch.mean(torch.abs(torch.gradient(x_tr, dim=2)[0]), dim=2), dim=2)

        x_alexnet = self.alexnet(x_tr)

        return x, x_alexnet, x_tv


def train(model, train_set, tv_gt, loss_weight_cr, loss_weight_class, loss_weight_tv, optimizer, criterion_cr, criterion_reg, criterion_tv, clip):
    model.train()

    epoch_loss = 0

    for ii, data in enumerate(train_set):
        src = data[0].to(device, torch.float)
        trg = data[1].to(device, torch.float)
        src = torch.abs(src[:, :, 8:-9])       # Reduce the length of the input signal to 896 so the AE can equalize the output signal length from encoder and decoder

        if trg.shape[0] != 32:
            tv_gt = torch.zeros((trg.shape[0], 2, 1)).to(device)

        optimizer.zero_grad()

        x, x_sense, x_tv = model(src)

        cr_loss = criterion_cr(x, src)
        sense_loss = criterion_reg(torch.squeeze(x_sense), trg)
        tv_loss = criterion_tv(x_tv, tv_gt)

        loss = loss_weight_cr * cr_loss + loss_weight_class * sense_loss + loss_weight_tv * tv_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_set)


if __name__ == '__main__':
    train_dataset = torch.load('../data/train_dataset_skin.pt', weights_only=False)

    CLIP = 1
    learning_rate = 1e-5
    loss_weight_cr = 0.3
    loss_weight_reg = 0.7
    loss_weight_tv = 1

    model = CR_NET(AE_in=2, AE_depth=6).to(device)
    model.clutter_gen.load_state_dict(torch.load("../models/clutter_extractor_skin.pkl", weights_only=False, map_location='cpu'))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_cr = nn.MSELoss()
    criterion_reg = nn.MSELoss()
    criterion_tv = nn.MSELoss()

    tv_gt_train = torch.zeros((32, 2, 1)).to(device)

    epoch_num = 10000
    train_loss = torch.zeros(epoch_num)

    min_loss = 1e9
    opt_epoch = 0
    opt_model = []
    for epoch in range(0, epoch_num):
        train_loss[epoch] = train(model, train_dataset, tv_gt_train, loss_weight_cr, loss_weight_reg, loss_weight_tv, optimizer, criterion_cr, criterion_reg, criterion_tv, CLIP)

        print(f"Training Loss at the {epoch}th epoch is {train_loss[epoch]}\n")
