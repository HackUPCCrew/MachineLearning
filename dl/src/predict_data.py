#!/usr/bin/env python3
import pickle
import keras
from keras.models import Sequential
from keras.layers.core import Dense
import numpy as np

LOAD_PICKLE = '/Users/krishnakalyan3/MOOC/MachineLearning/dl/src/model_trans_v1.pkl'
LOAD_MODEL = '/Users/krishnakalyan3/MOOC/MachineLearning/dl/src/crime_model.h5'


def model1(ip_shape):
    model = Sequential()
    model.add(Dense(31, init='uniform', activation='relu', input_shape=ip_shape[1:]))
    model.add(Dense(20, init='uniform', activation='relu'))
    model.add(Dense(11, init='uniform', activation='softmax'))
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
    return model


def load_pickle():
    trans_mode = pickle.load(open(LOAD_PICKLE,  'rb'), encoding='latin1')
    print('model Loaded')
    return trans_mode


def predict(weekday):
    gen_hrs = list(range(0, 24))
    model_dict = load_pickle()

    # Weekday
    get_w_lbl = model_dict['lbl_w'].transform([weekday])
    get_w_ohe = model_dict['ohe_w'].transform([get_w_lbl]).toarray()

    mini_batch = []
    # Hour
    for i in gen_hrs:
        get_h_ohe = model_dict['ohe_h'].transform(i).toarray()
        all_arr = np.hstack((get_w_ohe, get_h_ohe))
        mini_batch.append(all_arr)

    mini_batch = np.array(mini_batch).reshape(-1, 31)
    model = model1(mini_batch.shape)
    model.load_weights(LOAD_MODEL)
    yhats_train = model.predict(mini_batch, batch_size=24)
    max_y_value = np.argmax(yhats_train, axis=1)
    return max_y_value



if __name__ == '__main__':
    wd = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for i in wd:
        predict(i)
