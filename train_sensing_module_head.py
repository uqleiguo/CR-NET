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
from train_clutter_extractor_head import AE_Module
from sensing_module_head import SensingModule


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def signal_normalization(x, epsilon=1e-8):
    x_min = x.amin(dim=1, keepdim=True)
    x_max = x.amax(dim=1, keepdim=True)
    range_ = x_max - x_min

    x_norm = (x - x_min) / (range_ + epsilon)

    return x_norm


class CR_NET(nn.Module):
    def __init__(self, AE_in, AE_depth):
        super(CR_NET, self).__init__()

        self.clutter_gen = AE_Module(ch_in=AE_in, depth=AE_depth)
        self.channel_nums = [16, 32, 64, 128, 256]
        self.alexnet = SensingModule(self.channel_nums)

    def forward(self, x):
        x_input = x

        x = self.clutter_gen(x)

        x_tr_raw = torch.abs(x_input - x)
        bs, c, k1, k2 = x_tr_raw.shape
        x_tr_new = x_tr_raw.reshape(bs * c, k1 * k2)
        x_tr = signal_normalization(x_tr_new).view(bs, c, k1, k2).to(device)

        x_tv = torch.mean(torch.abs(torch.gradient(x_tr, dim=2)[0]), dim=2)

        x_sense = self.alexnet(x_tr)

        return x, x_sense, x_tv


def train(model, train_set, tv_gt, loss_weight_cr, loss_weight_sense, loss_weight_tv, optimizer, criterion_cr, criterion_class, criterion_tv, clip):
    model.train()

    epoch_loss = 0

    for ii, data in enumerate(train_set):
        src = data[0].to(device, torch.float)
        trg = data[1].to(device, torch.long)

        optimizer.zero_grad()

        x, x_class, x_tv = model(src)

        cr_loss = criterion_cr(x, src)
        sense_loss = criterion_class(x_class, trg)
        if x_tv.shape[0] == 16:
            tv_loss = criterion_tv(x_tv, tv_gt)
        else:
            tv_gt_tmp = torch.zeros((x_tv.shape[0], 2, 256)).to(device)
            tv_loss = criterion_tv(x_tv, tv_gt_tmp)

        loss = loss_weight_cr * cr_loss + loss_weight_sense * sense_loss + loss_weight_tv * tv_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_set)


if __name__ == '__main__':
    train_dataset = torch.load('../data/train_dataset_head.pt', weights_only=False)

    CLIP = 1
    learning_rate = 1e-6
    loss_weight_cr = 0.3
    loss_weight_sense = 0.7
    loss_weight_tv = 1

    model = CR_NET(AE_in=2, AE_depth=6).to(device)
    model.clutter_gen.load_state_dict(torch.load("../models/clutter_extractor_head.pkl", weights_only=False, map_location='cpu'))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_cr = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()
    criterion_tv = nn.MSELoss()

    tv_gt_train = torch.zeros((16, 2, 256)).to(device)

    epoch_num = 600
    train_loss = torch.zeros(epoch_num)

    for epoch in range(0, epoch_num):
        train_loss[epoch] = train(model, train_dataset, tv_gt_train, loss_weight_cr, loss_weight_sense, loss_weight_tv, optimizer, criterion_cr, criterion_class, criterion_tv, CLIP)

        print(f"Train Loss at the {epoch}th epoch is {train_loss[epoch]}\n")


    print('debug')
