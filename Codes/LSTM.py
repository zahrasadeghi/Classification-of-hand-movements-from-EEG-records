import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

data = pd.read_csv('./train/subj1_series1_data.csv', skiprows=[0], header=None)
label = pd.read_csv('./train/subj1_series1_events.csv', skiprows=[0], header=None)
features = data.iloc[:, 1:]
labels = label.iloc[:, 1:7]
y_finl = label.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(features, y_finl, test_size=0.5, shuffle=False, random_state=42)
print(features.head())

epochs = 1
n_classes = 1
n_units = 200
n_features = 32
batch_size = 35

xplaceholder = tf.placeholder('float', [None, n_features])
yplaceholder = tf.placeholder('float')


def recurrent_neural_network_model():
    layer = {'weights': tf.Variable(tf.random_normal([n_units, n_classes])),
             'bias': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.split(xplaceholder, n_features, 1)
    print(x)

    lstm_cell = rnn.BasicLSTMCell(n_units)

    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['bias']

    return output


def train_neural_network():
    logit = recurrent_neural_network_model()
    logit = tf.reshape(logit, [-1])

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=yplaceholder))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        for epoch in range(epochs):
            epoch_loss = 0

            i = 0
            for i in range(int(len(X_train) / batch_size)):
                start = i
                end = i + batch_size

                batch_x = np.array(X_train[start:end])
                batch_y = np.array(y_train[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={xplaceholder: batch_x, yplaceholder: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch, 'completed out of', epochs, 'loss:', epoch_loss)

        pred = tf.round(tf.nn.sigmoid(logit)).eval({xplaceholder: np.array(X_test), yplaceholder: np.array(y_test)})
        f1 = f1_score(np.array(y_test), pred, average='macro')
        accuracy = accuracy_score(np.array(y_test), pred)
        recall = recall_score(y_true=np.array(y_test), y_pred=pred)
        precision = precision_score(y_true=np.array(y_test), y_pred=pred)
        print("F1 Score:", f1)
        print("Accuracy Score:", accuracy)
        print("Recall:", recall)
        print("Precision:", precision)


train_neural_network()
