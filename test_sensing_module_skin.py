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
from train_sensing_module_skin import CR_NET


matplotlib.use('TkAgg')

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

test_dataset = torch.load('../data/test_dataset_skin.pt', weights_only=False)


def evaluate(model, test_set, tv_gt, loss_weight_cr, loss_weight_class, loss_weight_tv, criterion_cr, criterion_reg, criterion_tv):
    model.eval()

    epoch_loss = 0
    epoch_reg_loss = 0
    epoch_tv_loss = 0
    gt_label_array = np.zeros((300, 1))
    predict_thickness = np.zeros((300, 1))
    src_signal = np.zeros((300, 2, 384))
    clutter_signal = np.zeros((300, 2, 384))

    with torch.no_grad():
        for ii, data in enumerate(test_set):
            src = data[0].to(device, torch.float)
            trg = data[1].to(device, torch.float)
            src = torch.abs(src[:, :, 8:-9])  # Reduce the length of the input signal to 896 so the AE can equalize the output signal length from encoder and decoder

            x, x_sense, x_tv = model(src)

            cr_loss = criterion_cr(x, src)
            sense_loss = criterion_reg(x_sense, trg)
            tv_loss = criterion_tv(x_tv, tv_gt)

            loss = loss_weight_cr * cr_loss + loss_weight_class * sense_loss + loss_weight_tv * tv_loss

            gt_label_array[ii] = np.array(trg.cpu())
            predict_thickness[ii] = np.array(x_sense.cpu())

            epoch_loss += loss
            epoch_reg_loss += sense_loss
            epoch_tv_loss += tv_loss

            src_signal[ii, :, :] = src.detach().cpu()
            clutter_signal[ii, :, :] = x.detach().cpu()


        return epoch_loss / len(test_dataset), epoch_reg_loss / len(test_dataset), epoch_tv_loss / len(test_dataset), predict_thickness, gt_label_array, src_signal, clutter_signal


model = CR_NET(AE_in=2, AE_depth=6).to(device)
criterion_cr = nn.MSELoss()
criterion_reg = nn.MSELoss()
criterion_tv = nn.MSELoss()
loss_weight_cr = 0.2
loss_weight_class = 0.8
loss_weight_tv = 1

tv_gt_test = torch.zeros((1, 2, 1)).to(device)

gt_sort = np.zeros((300, 10))
predict_sort = np.zeros((300, 10))
for nn in range(10):
    model.load_state_dict(torch.load("../models/cr_net_skin_R" + str(nn + 1) + '.pkl', weights_only=False, map_location='cpu'))
    _, reg_loss, _, predict, gt, _, _ = evaluate(model, test_dataset, tv_gt_test, loss_weight_cr, loss_weight_class, loss_weight_tv, criterion_cr, criterion_reg, criterion_tv)

    gt_sort_idx = np.argsort(gt[:, 0])
    gt_sort[:, nn] = gt[gt_sort_idx, 0]
    predict_sort[:, nn] = predict[gt_sort_idx, 0]


plt.figure()
plt.plot(gt_sort * 1e2, gt_sort * 1e2, 'r--', lw=3)
plt.scatter(gt_sort * 1e2, predict_sort * 1e2, color='blue')
plt.show()
