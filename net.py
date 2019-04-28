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
    def __init__(self, title):
        self.title = title

        self.model = keras.Sequential([
            keras.layers.Dense(145, activation='relu'),
            keras.layers.Dropout(0.3, noise_shape=None, seed=None),
            keras.layers.Dense(145, activation='relu'),
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
    learner = NetLearner("test1")

    truth = pd.read_csv("truth.csv")
    false = pd.read_csv("false.csv")

    truth_y = np.ones(len(truth.iloc[:, 0]))
    false_y = np.zeros(len(false.iloc[:, 0]))

    X_train1, X_test1, Y_train1, Y_test1 = train_test_split(np.array(truth), np.array(truth_y))
    X_train2, X_test2, Y_train2, Y_test2 = train_test_split(np.array(false), np.array(false_y))

    X_train = np.concatenate((X_train1, X_train2)).astype(np.float32)
    Y_train = np.concatenate((Y_train1, Y_train2)).astype(np.float32)
    X_test = np.concatenate((X_test1, X_test2)).astype(np.float32)
    Y_test = np.concatenate((Y_test1, Y_test2)).astype(np.float32)

    learner.learn(X_train, Y_train)

    learner.print_accuracy(X_test, Y_test)
