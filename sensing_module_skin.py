import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use('TkAgg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### Use four cascaded 1D convolution layers followed by two linear layers
class AlexNet(nn.Module):
    def __init__(self, channel_num):
        super(AlexNet, self).__init__()

        self.four_conv1d = nn.Sequential(
            nn.Conv1d(2, channel_num[0], kernel_size=11, stride=4, padding=2),
            nn.BatchNorm1d(channel_num[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),

            nn.Conv1d(channel_num[0], channel_num[1], kernel_size=5, padding=2),
            nn.BatchNorm1d(channel_num[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),

            nn.Conv1d(channel_num[1], channel_num[2], kernel_size=3, padding=1),
            nn.BatchNorm1d(channel_num[2]),
            nn.ReLU(),

            nn.Conv1d(channel_num[2], channel_num[3], kernel_size=3, padding=1),
            nn.BatchNorm1d(channel_num[3]),
            nn.ReLU(),

            nn.Conv1d(channel_num[3], channel_num[4], kernel_size=3, padding=1),
            nn.BatchNorm1d(channel_num[4]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.linear_layer = nn.Sequential(
            nn.Linear(2816, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, x):

        x = self.four_conv1d(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer(x.squeeze())

        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, train_dataset, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for ii, data in enumerate(train_dataset):
        src = data[0].to(device, dtype=torch.float)
        trg = data[1].to(device, dtype=torch.float)
        src = torch.abs(src[:, :, 8:-9])

        optimizer.zero_grad()

        y = model(src)

        loss = criterion(torch.squeeze(y), trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_dataset)


def evaluate(model, test_dataset, criterion):
    model.eval()

    epoch_loss = 0
    gt_label_array = np.zeros((300, 1))

    with torch.no_grad():
        for ii, data in enumerate(test_dataset):
            src = data[0].to(device, dtype=torch.float)
            trg = data[1].to(device, dtype=torch.float)
            src = torch.abs(src[:, :, 8:-9])

            y = model(src)

            loss = criterion(torch.squeeze(y, dim=0), trg)

            gt_label_array[ii] = np.array(trg.cpu())

            epoch_loss += loss

    # good_cases_label_array = np.zeros((99, 1))
    # for ii in range(len(good_cases_label)):
    #     good_cases_label_array[ii] = good_cases_label[ii]

    return epoch_loss / len(test_dataset)


if __name__ == '__main__':
    train_dataset = torch.load('../data/train_test_data/train_exp_diff_thick_regression_raw_29_07.pt', weights_only=False)
    test_dataset = torch.load('../data/train_test_data/test_exp_diff_thick_regression_raw_29_07.pt', weights_only=False)

    channel_nums = [16, 32, 64, 128, 256]
    model = AlexNet(channel_nums).to(device)
    # model.load_state_dict(torch.load('../models/SkinNet_ResNet_model_1275_v2.pkl', map_location='cpu'))
    criterion = nn.MSELoss()
    CLIP = 1
    learning_rate = 1e-5
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    epoch_num = 10000
    train_loss = torch.zeros(epoch_num)
    test_loss = torch.zeros(epoch_num)

    min_loss = 1e9
    for epoch in range(0, epoch_num):

        train_loss[epoch] = train(model, train_dataset, optimizer, criterion, CLIP)

        test_loss[epoch] = evaluate(model, test_dataset, criterion)

        print(f"Train Loss at the {epoch}th epoch is {train_loss[epoch]} and Test Loss is {test_loss[epoch]}\n")

        if min_loss > test_loss[epoch]:
            min_loss = test_loss[epoch]
            # opt_model = model.state_dict()
            torch.save(model.state_dict(), '../models/skin_models/SkinNet_cnn_model_diff_thick_test_regression_raw_29_07_10000epoch_R1.pkl')

    test_loss = test_loss.detach().cpu().numpy()
    train_loss = train_loss.detach().cpu().numpy()
    np.save('../results/skin_results/test_loss_diff_thick_cnn_raw_29_07_10000epoch_R1', test_loss)
    np.save('../results/skin_results/train_loss_diff_thick_cnn_raw_29_07_10000epoch_R1', train_loss)

    print('debug')






