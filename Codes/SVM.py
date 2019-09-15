import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score as auc
from sklearn import svm

#############function to read data###########

def prepare_data_train(fname):
    """ read and prepare training data """
    # Read data
    data = pd.read_csv(fname)
    # events file
    events_fname = fname.replace('_data', '_events')
    # read event file
    labels = pd.read_csv(events_fname)
    clean = data.drop(['id'], axis=1)  # remove id
    labels = labels.drop(['id'], axis=1)  # remove id
    return clean, labels


def prepare_data_test(fname):
    """ read and prepare test data """
    # Read data
    data = pd.read_csv(fname)
    return data


def data_preprocess_train(X):
    X_prep = scaler.fit_transform(X)
    # do here your preprocessing
    return X_prep


def data_preprocess_test(X):
    X_prep = scaler.transform(X)
    # do here your preprocessing
    return X_prep


def read_data(subjects):
    ###loop on subjects and 8 series for train data + 2 series for test data
    y_raw = []
    raw = []
    for subject in subjects:
        ################ READ DATA ################################################
        fnames = glob('../train/subj%d_series*_data.csv' % (subject))
        for fname in fnames:
            data, labels = prepare_data_train(fname)
            raw.append(data)
            y_raw.append(labels)
    X = pd.concat(raw)
    y = pd.concat(y_raw)
    # transform in numpy array
    # transform train data in numpy array
    X_train = np.asarray(X.astype(float))
    y = np.asarray(y.astype(float))
    X_train, X_test, y, y_test = train_test_split(X_train, y, test_size=0.20)
    return X_train, X_test, y, y_test


def train(X_train, X_test, y, y_test):
    clf = svm.SVC(probability=True)
    pred = np.empty((X_test.shape[0], 6))
    X_train = data_preprocess_train(X_train)
    X_test = data_preprocess_test(X_test)

    y_label = np.empty((y.shape[0],))
    for n in range(0, y.shape[0] - 1):
        if max(y[n, :]) == 0:
            y_label[n] = 0
        else:
            y_label[n] = y[n, :].nonzero()[0][0] + 1

    clf.fit(X_train[::subsample, :], y_label[::subsample])
    y_pred = clf.predict(X_test)
    print(len(y_pred))
    print(y_pred[0])
    from sklearn.metrics import classification_report, confusion_matrix

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    pred = clf.predict_proba(X_test)[:, 1:]

    print(pred.shape)
    print(pred)
    print(y_label)
    return pred



def valscore(pred, y_test):
    targs = y_test
    # print(targs[0])
    # print(y_test[0])
    # bool_preds = (pred > 0.3).astype(np.int32)
    # print(classification_report(y_test, bool_preds))
    print("auc:", np.mean([auc(targs[:, j], pred[:, j]) for j in range(6)]))
    # print("precision:", np.mean([precision_score(targs[:, j], bool_preds[:, j]) for j in range(6)]))
    # print("Recall:", np.mean([recall_score(y_true=targs[:, j], y_pred=bool_preds[:, j]) for j in range(6)]))
    # print("F1 Score:", np.mean([f1_score(targs[:, j], bool_preds[:, j], average='macro') for j in range(6)]))
    # print("Accuracy Score:", np.mean([accuracy_score(targs[:, j], bool_preds[:, j]) for j in range(6)]))


scaler = StandardScaler()

# training subsample.if you want to downsample the training data
subsample = 10
subjects = range(1, 9)
X_train, X_test, y, y_test = read_data(subjects)
pred = train(X_train, X_test, y, y_test)
valscore(pred, y_test)


