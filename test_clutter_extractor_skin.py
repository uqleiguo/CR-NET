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
from train_clutter_extractor_skin import AE_Module


matplotlib.use('TkAgg')

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


test_dataset = torch.load('../data/test_dataset_skin.pt', weights_only=False)


def evaluate(model, test_set, criterion):
    model.eval()

    epoch_loss = 0
    gt = np.zeros((300, 2, 384))
    gen = np.zeros((300, 2, 384))

    with torch.no_grad():
        for ii, data in enumerate(test_set):
            src = data[0].to(device, torch.float)
            src = torch.abs(src[:, :, 8:-9])            # Reduce the length of the input signal to 896 so the AE can equalize the output signal length from encoder and decoder

            x = model(src)

            loss = criterion(x, src)

            epoch_loss += loss

            gt[ii, :, :] = src.cpu()
            gen[ii, :, :] = x.cpu()

        return epoch_loss / len(test_dataset), gt, gen


model = AE_Module(ch_in=2, depth=6).to(device)
criterion = nn.MSELoss()

model.load_state_dict(torch.load("../models/clutter_extractor_skin.pkl", weights_only=False, map_location='cpu'))
test_loss, gt, gen = evaluate(model, test_dataset, criterion)

matplotlib.use('TkAgg')
plt.plot(gt[23, 0, :])
plt.plot(gen[23, 0, :])
plt.show()
