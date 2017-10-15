#!/usr/bin/env python3

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten

class model():
    def __init__(self):
        self.learning_rate = 0.001

    def model1(self):
        model = Sequential()
        model.add(Dense(30, init='uniform', activation='relu'))
        model.add(Dense(20, init='uniform', activation='relu'))
        model.add(Dense(10, init='uniform', activation='sigmoid'))
        adam = keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
        return model
