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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import bootstrap
from train_sensing_module_head import CR_NET
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use('TkAgg')

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

test_dataset = torch.load('../data/test_dataset_head.pt', weights_only=False)

CASE_NUM = len(test_dataset)


def evaluate(model, test_set, tv_gt, loss_weight_cr, loss_weight_class, loss_weight_tv, criterion_cr, criterion_class, criterion_tv):
    model.eval()

    epoch_loss = 0
    epoch_class_loss = 0
    epoch_tv_loss = 0
    gt_label_array = np.zeros((CASE_NUM, 1))
    predict_class = np.zeros((CASE_NUM, 1))
    acc_num = 0
    predict_scores = np.zeros((CASE_NUM, 2))
    clutter_signal = np.zeros((CASE_NUM, 2, 256, 256))
    src_signal = np.zeros((CASE_NUM, 2, 256, 256))

    with torch.no_grad():
        for ii, data in enumerate(test_set):
            src = data[0].to(device, torch.float)
            trg = data[1].to(device, torch.long)

            x, x_class, x_tv = model(src)

            if torch.argmax(x_class) == trg:
                acc_num += 1

            predict_class[ii] = np.array(torch.argmax(x_class).detach().cpu())
            predict_scores[ii, :] = x_class.detach().cpu().numpy()

            cr_loss = criterion_cr(x, src)
            sense_loss = criterion_class(x_class, torch.squeeze(trg))
            tv_loss = criterion_tv(x_tv, tv_gt)

            loss = loss_weight_cr * cr_loss + loss_weight_class * sense_loss + loss_weight_tv * tv_loss

            gt_label_array[ii] = np.array(trg.cpu())

            epoch_loss += loss.item()
            epoch_class_loss += sense_loss.item()
            epoch_tv_loss += tv_loss.item()

        return acc_num / len(test_dataset), epoch_class_loss / len(test_dataset), gt_label_array, predict_class, predict_scores, clutter_signal, src_signal


model = CR_NET(AE_in=2, AE_depth=6).to(device)
criterion_cr = nn.MSELoss()
criterion_class = nn.CrossEntropyLoss()
criterion_tv = nn.MSELoss()
loss_weight_cr = 0.5
loss_weight_class = 0.5
loss_weight_tv = 1
tv_gt_test = torch.zeros((1, 2, 256)).to(device)

model_num = 20

gt_array = np.zeros((CASE_NUM, model_num))
predict_class_array = np.zeros((CASE_NUM, model_num))
predict_score_array = np.zeros((CASE_NUM, 2, model_num))
acc_rate = np.zeros((model_num, 1))
precision_array = np.zeros((model_num, 1))
recall_array = np.zeros((model_num, 1))
f1_array = np.zeros((model_num, 1))
cm = np.zeros((2, 2, model_num))
clutter_signal_array = np.zeros((CASE_NUM, 2, 256, 256, model_num))
src_signal_array = np.zeros((CASE_NUM, 2, 256, 256, model_num))

for nn in range(model_num):
    model.load_state_dict(torch.load("../models/cr_net_head_R" + str(nn + 1) + '.pkl', weights_only=False, map_location='cpu'))
    _, _, gt_labels_tmp, predict_class_tmp, predict_score_tmp, clutter_signal, src_signal = evaluate(model, test_dataset, tv_gt_test, loss_weight_cr, loss_weight_class, loss_weight_tv, criterion_cr, criterion_class, criterion_tv)
    gt_array[:, nn] = np.squeeze(gt_labels_tmp)
    predict_class_array[:, nn] = np.squeeze(predict_class_tmp)
    predict_score_array[:, :, nn] = predict_score_tmp
    clutter_signal_array[:, :, :, :, nn] = clutter_signal
    src_signal_array[:, :, :, :, nn] = src_signal
    cm[:, :, nn] = confusion_matrix(gt_array[:, nn], predict_class_array[:, nn])

    tn, fp, fn, tp = confusion_matrix(gt_array[:, nn], predict_class_array[:, nn]).ravel()

    acc_rate[nn, 0] = (tp + tn) / (tp + tn + fp + fn)
    precision_array[nn, 0] = tp / (tp + fp)
    recall_array[nn, 0] = tp / (tp + fn)
    f1_array[nn, 0] = (2 * tp) / (2 * tp + fp + fn)


res_acc = bootstrap((acc_rate,), np.mean, confidence_level=0.95, n_resamples=10000, method='percentile', random_state=27)
res_precision = bootstrap((precision_array,), np.mean, confidence_level=0.95, n_resamples=10000, method='percentile', random_state=27)
res_recall = bootstrap((recall_array,), np.mean, confidence_level=0.95, n_resamples=10000, method='percentile', random_state=27)
res_f1 = bootstrap((f1_array,), np.mean, confidence_level=0.95, n_resamples=10000, method='percentile', random_state=27)

conf95_acc = res_acc.confidence_interval
conf95_precision = res_precision.confidence_interval
conf95_recall = res_recall.confidence_interval
conf95_f1 = res_f1.confidence_interval

print(f'The 95% confidence interval of accuracy is {conf95_acc[0]}~{conf95_acc[1]}, the mean is {np.mean(acc_rate)}, and the standard deviation is {np.std(acc_rate)}')
print(f'The 95% confidence interval of precision is {conf95_precision[0]}~{conf95_precision[1]}, the mean is {np.mean(precision_array)}, and the standard deviation is {np.std(precision_array)}')
print(f'The 95% confidence interval of recall is {conf95_recall[0]}~{conf95_recall[1]}, the mean is {np.mean(recall_array)}, and the standard deviation is {np.std(recall_array)}')
print(f'The 95% confidence interval of f1 score is {conf95_f1[0]}~{conf95_f1[1]}, the mean is {np.mean(f1_array)}, and the standard deviation is {np.std(f1_array)}')


cm = np.mean(cm, axis=2) / 24
disp = ConfusionMatrixDisplay(cm, display_labels=['IS', 'HS'])

fig, ax = plt.subplots(figsize=(8.2, 8))
disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='.4f')
disp.ax_.set_xlabel('Predicted label', fontsize=28)
disp.ax_.set_ylabel('True label', fontsize=28)
disp.ax_.tick_params(axis='both', labelsize=28)

for text in disp.text_.ravel():
    text.set_fontsize(28)

plt.show()
