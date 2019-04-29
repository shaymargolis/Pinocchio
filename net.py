from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split

class NetLearner:
    def __init__(self, title, length):
        self.title = title

        self.model = keras.Sequential([
            keras.layers.Dense(length, activation='relu'),
            keras.layers.Dropout(0.3, noise_shape=None, seed=None),
            keras.layers.Dense(length, activation='relu'),
            keras.layers.Dropout(0.3, noise_shape=None, seed=None),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    def learn(self, X_train, Y_train):
        self.model.fit(X_train, Y_train, epochs=5)
        self.model.save("Model/" + self.title + ".h5")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def print_accuracy(self, X_test, Y_test):
        test_loss, test_acc = self.model.evaluate(X_test, Y_test)
        print('Test accuracy:', test_acc)

if __name__ == "__main__":
    learner = NetLearner("test1", 145)

    truth = pd.read_csv("truth.csv")
    false = pd.read_csv("false.csv")

    truth = np.array(truth)
    false = np.array(false)

    odd = range(1, 144, 2)
    even = range(0, 144, 2)

    for i in range(len(truth)):
        truth[i, even] -= truth[i, 16]
        truth[i, odd] -= truth[i, 17]

    for i in range(len(false)):
        false[i, even] -= false[i, 16]
        false[i, odd] -= false[i, 17]

    print(truth[:, [16, 17]])

    truth_y = np.ones(len(truth[:, 0]))
    false_y = np.zeros(len(false[:, 0]))

    data = np.concatenate((truth, false))
    y = np.concatenate((truth_y, false_y))

    print(y)

    X_train, X_test, Y_train, Y_test = train_test_split(data, y)

    learner.learn(X_train, Y_train)

    learner.print_accuracy(X_test, Y_test)
