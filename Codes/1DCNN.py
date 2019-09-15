import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import numpy as np
import pandas as pd


class eegdata(Dataset):
    def __init__(self, path, validation=False, subjects=range(1, 10)):
        super().__init__()
        data, self.target = self.readfiles(path, validation, subjects)
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        self.data = self.norm(data)

    def norm(self, x):
        return ((x - self.mean) / self.std)

    def to_np(self, values):

        # get total lines of data count
        count = 0
        for i in range(len(values)):
            count += len(values[i])

        # create np array size of all data
        ret = np.zeros((count, len(values[0][0])))

        # copy data into np array
        ix = 0
        for i in range(len(values)):
            ret[ix:ix + len(values[i]), :] = values[i]
            ix += len(values[i])
        return ret

    def readfiles(self, path, validation, subjects):

        allx = []
        ally = []

        if not validation:
            series = [1, 2, 4, 5, 6, 7, 8]
        else:
            series = [3]

        for i in subjects:
            print('log: reading subject {}...'.format(i))
            xs = None
            ys = None
            for j in series:
                data = 'subj{}_series{}_data.csv'.format(i, j)
                events = 'subj{}_series{}_events.csv'.format(i, j)

                x = pd.read_csv(path + data).values[:, 1:]
                xs = x if xs is None else np.vstack((xs, x))

                y = pd.read_csv(path + events).values[:, 1:]
                ys = y if ys is None else np.vstack((ys, y))

            allx.append(xs)
            ally.append(ys)

        xs = self.to_np(allx)
        ys = self.to_np(ally)

        return xs, ys

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)


# convolutional network model we will train to detect patterns in readings.
class convmodel(nn.Module):
    def __init__(self, out_classes, drop=0.5, d_linear=124):
        super().__init__()

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=0, stride=1)
        self.bn = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2, stride=2)
        self.linear1 = nn.Linear(8128, d_linear)

        self.linear3 = nn.Linear(d_linear, out_classes)
        self.dropout1 = nn.Dropout(drop)
        self.dropout2 = nn.Dropout(drop)
        self.dropout3 = nn.Dropout(drop)

        self.conv = nn.Sequential(self.conv2, nn.ReLU(inplace=True), self.bn, \
                                  self.pool, self.dropout1)
        self.dense = nn.Sequential(self.linear1, nn.ReLU(inplace=True), self.dropout2, \
                                   self.dropout3, self.linear3)

    def forward(self, x):
        bs = x.size(0)
        x = self.conv(x)
        x = x.view(bs, -1)
        output = self.dense(x)

        return torch.sigmoid(output)


def showDataFigure(mse):
    x2 = []
    y2 = []
    for i in mse:
        x2.append(i[0])
        y2.append(i[1])
    plt.plot(x2, y2)
    plt.show()


# also create individual datasets for subjects


# Batch creator. When training it will return random locations in the dataset. The data is a time series
# and so we feed previous readings (going back window_size) in with each index. Rather than feed in all window_size
# previous readings, we subsample and take every 4th set of readings.
def get_batch(dataset, batch_size=2000, val=False, index=None):
    if val == False:
        index = random.randint(window_size, len(dataset) - 16 * batch_size)
        indexes = np.arange(index, index + 16 * batch_size, 16)

    else:
        indexes = np.arange(index, index + batch_size)

    batch = np.zeros((batch_size, num_features, window_size // 4))

    b = 0
    for i in indexes:
        start = i - window_size if i - window_size > 0 else 0

        tmp = dataset.data[start:i]
        batch[b, :, :] = tmp[::4].transpose()

        b += 1

    targets = dataset.target[indexes]
    return torch.DoubleTensor(batch), torch.DoubleTensor(targets)


def train(traindata, epochs, printevery=1, shuffle=True):
    model.train()
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for i in range(1000):
            print(i)
            optim.zero_grad()
            x, y = get_batch(traindata)
            preds = model(x)
            loss = F.binary_cross_entropy(preds.view(-1), y.view(-1))
            loss.backward()
            total_loss += loss.data

            optim.step()

            if (i + 1) % printevery == 0:
                print("epoch: %d, iter %d/%d, loss %.4f" % (
                    epoch + 1, i + 1, 2000, total_loss / printevery), end='images/\r')
                losses.append((i, total_loss / printevery))
                total_loss = 0
    return losses


def getPredictions(data, test):
    model.eval()
    p = []
    res = []
    i = window_size
    bs = 2000
    while i < len(data):
        if i + bs > len(data):
            bs = len(data) - i

        x, y = get_batch(data, bs, index=i, val=True)
        x = (x)

        preds = model(x)
        preds = preds.squeeze(1)

        p.append(np.array(preds.data))
        res.append(np.array(y.data))

        i += bs

    preds = p[0]
    for i in p[1:]:
        preds = np.vstack((preds, i))

    targs = res[0]
    for i in res[1:]:
        targs = np.vstack((targs, i))

    return preds, targs


def valscore(data, test=False):
    preds, targs = getPredictions(data, test)
    bool_preds = preds > 0.3
    print("auc:", np.mean([auc(targs[:, j], preds[:, j]) for j in range(6)]))
    print("precision:", np.mean([precision_score(targs[:, j], bool_preds[:, j]) for j in range(6)]))
    print("Recall:", np.mean([recall_score(y_true=targs[:, j], y_pred=bool_preds[:, j]) for j in range(6)]))
    print("F1 Score:", np.mean([f1_score(targs[:, j], bool_preds[:, j], average='macro') for j in range(6)]))
    print("Accuracy Score:", np.mean([accuracy_score(targs[:, j], bool_preds[:, j]) for j in range(6)]))


# load all subjects data into one big array using object created above
traindata = eegdata('train/')
valdata = eegdata('train/', validation=True)

print("reading files done!")

# some parameteres for the model
num_features = 32
window_size = 1024
batch_size = 2000

model = convmodel(6).double()

optim = torch.optim.Adadelta(model.parameters(), lr=1, eps=1e-10)

bs = batch_size

# train model for one epoch
losses = train(traindata, 1)

# save general model
torch.save(model.state_dict(), "CNN1D_1000E")

# load model
model = convmodel(6).double()
model.load_state_dict(torch.load("CNN1D_1000E"))


# see how we scored on validation set
valscore(valdata)
showDataFigure(losses)
