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
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from train_clutter_extractor_head import AE_Module


matplotlib.use('TkAgg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_dataset = torch.load('../data/test_dataset_head.pt', weights_only=False)


def evaluate(model, test_set, criterion):
    model.eval()

    epoch_loss = 0
    gt = np.zeros((58, 2, 256, 256))
    gen = np.zeros((58, 2, 256, 256))

    with torch.no_grad():
        for ii, data in enumerate(test_set):
            src = torch.abs(data[0]).to(device, torch.float)

            x = model(src)

            loss = criterion(x, src)

            epoch_loss += loss

            gt[ii, :, :] = src.cpu()
            gen[ii, :, :] = x.cpu()

        return epoch_loss / len(test_dataset), gt, gen


model = AE_Module(ch_in=2, depth=6).to(device)
criterion = nn.MSELoss()

model.load_state_dict(torch.load("../models/clutter_extractor_head.pkl", weights_only=False, map_location='cpu'))
test_loss, gt, gen = evaluate(model, test_dataset, criterion)

matplotlib.use('TkAgg')
plt.imshow(gt[0, 0, :, :], cmap='jet', vmin=0, vmax=1)
plt.figure()
plt.imshow(gen[0, 0, :, :], cmap='jet', vmin=0, vmax=1)
plt.show()

