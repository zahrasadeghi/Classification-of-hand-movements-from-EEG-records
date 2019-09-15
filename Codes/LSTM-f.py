import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import random
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense

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

        series = [1, 2, 4, 5, 6, 7, 8] if validation == False else [3]

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

traindata = eegdata('train/', subjects=[1, 2])
valdata = eegdata('train/', validation=True)

num_features = 32
window_size = 1024
batch_size = 2000
num_classes = 6

# LSTM hyperparmaters
num_layers = 3
hidden_dim = 32


def get_batch(dataset, batch_size=2000, val=False, index=None, single_label=False):
    if val == False:
        index = random.randint(window_size, len(dataset) - 16 * batch_size)
        indexes = np.arange(index, index + 16 * batch_size, 16)

    else:
        indexes = np.arange(index, index + batch_size)

    batch = np.zeros((batch_size, window_size // 4, num_features))

    b = 0
    targets = np.zeros((batch_size, window_size // 4, num_classes))

    for i in indexes:
        start = i - window_size if i - window_size > 0 else 0

        tmp = dataset.data[start:i]
        tmp_tgts = dataset.target[start:i]

        batch[b, :, :] = tmp[::4]
        targets[b, :] = tmp_tgts[::4]

        b += 1

    if single_label:
        targets = dataset.target[indexes]

    return batch, targets

model = Sequential()
model.add(LSTM(hidden_dim, return_sequences=True, input_shape=(window_size//4, num_features)))
for _ in range(num_layers-1):
  model.add(LSTM(hidden_dim, dropout=0.3, return_sequences=True))

model.add(TimeDistributed(Dense(num_classes, activation="softmax")))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()

from keras.utils import Sequence


class TrainBatchGenerater(Sequence):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset) // batch_size

    def __getitem__(self, idx):
        return get_batch(self.dataset)


class ValidBatchGenerater(Sequence):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return (len(self.dataset) // batch_size) - 1

    def __getitem__(self, idx):
        index = idx * batch_size + window_size
        my_batch_size = batch_size
        if index > len(self.dataset):
            my_batch_size = len(self.dataset) - batch_size

        return get_batch(self.dataset, my_batch_size, index=index, val=True)

train_generator = TrainBatchGenerater(traindata)
valid_generator = ValidBatchGenerater(valdata)
# Training parameters
num_epochs = 10

history = model.fit_generator(
    train_generator,
    validation_data=valid_generator,
    epochs=num_epochs
)

# visualize_loss_and_acc(history)