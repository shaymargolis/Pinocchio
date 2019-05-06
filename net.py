from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

class NetLearner:
    def __init__(self, title, length):
        self.title = title

        #model.add(Dense(units=20, activation='relu', kernel_regularizer='l2', input_dim=x.shape[1]))
        # model.add(Dense(units=10, activation='relu', kernel_regularizer='l2'))

        self.model = keras.Sequential([
            keras.layers.Dense(20, input_dim=length, activation='relu', kernel_regularizer='l2'),
            keras.layers.Dropout(0.3, noise_shape=None, seed=None),
            keras.layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            keras.layers.Dropout(0.2, noise_shape=None, seed=None),
            keras.layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            keras.layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            keras.layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            keras.layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            keras.layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            keras.layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            keras.layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            keras.layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            keras.layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            keras.layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            keras.layers.Dense(1, activation='sigmoid')
            #keras.layers.Dense(10, input_dim=length, activation='linear'),
            #keras.layers.Dense(2, input_dim=length, activation='sigmoid')
        ])

        # self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def learn(self, X_train, Y_train):
        self.model.fit(X_train, Y_train, epochs=1000, batch_size=20)
        self.model.save("Model/" + self.title + ".h5")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def print_accuracy(self, X_test, Y_test):
        test_loss, test_acc = self.model.evaluate(X_test, Y_test)
        print('Test accuracy:', test_acc)

if __name__ == "__main__":
    learner = NetLearner("test1", 137)

    truth = pd.read_csv("truth.csv")
    false = pd.read_csv("false.csv")

    print(len(truth))

    truth["Type"] = "T"
    false["Type"] = "F"

    truth = truth.append(false)


    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(truth["Type"])
    encoded_Y = encoder.transform(truth["Type"])
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)

    print(dummy_y)

    truth = truth.drop(["Unnamed: 0", "Type", '136', '137', '138', '139', '140', '141', '142'], axis=1)

    print(truth)
    print(len(truth.columns))

    X_train, X_test, Y_train, Y_test = train_test_split(np.array(truth), dummy_y)

    learner.learn(X_train, Y_train)

    learner.print_accuracy(X_test, Y_test)
