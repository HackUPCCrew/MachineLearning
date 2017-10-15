#!/usr/bin/env python3
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils

DATA_PATH = '/Users/krishnakalyan3/MOOC/MachineLearning/data/crime_data.csv'
DATA_TRAIN_POINTS = 40000


def model1(ip_shape):
    model = Sequential()
    model.add(Dense(31, init='uniform', activation='relu', input_shape=ip_shape[1:]))
    model.add(Dense(20, init='uniform', activation='relu'))
    model.add(Dense(11, init='uniform', activation='softmax'))
    adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
    return model

def save_model(data_dict):
    import pickle
    result_filename = ''.join(['model_trans_', 'v1', '.pkl'])
    pickle.dump(data_dict, open(result_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    crime_data = pd.read_csv(DATA_PATH)
    crime_data = crime_data[:400000]

    # hours, weekend, k_means.cluster
    crime_sub = crime_data.iloc[:, 4:7]


    filter_data = []
    for i in range(1, 11):
        if i == 4:
            pass

        current_data = crime_sub[crime_sub['k_means.cluster'] == i][:40000]
        filter_data.append(current_data)

    final_data = pd.concat(filter_data)

    lbl_w = LabelEncoder()
    lbl_w_fit = lbl_w.fit(final_data['weekend'])
    le_weekend = lbl_w_fit.transform(final_data['weekend']).reshape(-1, 1)

    enc_w = OneHotEncoder()
    enc_w_fit = enc_w.fit(le_weekend)
    en_w = enc_w_fit.transform(le_weekend).toarray()

    enc_h = OneHotEncoder()
    enc_h_fit = enc_h.fit(final_data['hours'].reshape(-1, 1))
    en_h = enc_h_fit.transform(final_data['hours'].reshape(-1, 1)).toarray()

    y_all = np_utils.to_categorical(final_data['k_means.cluster'], 11)
    x_all = np.hstack((en_w, en_h))

    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=0)

    ip_shape = x_train.shape
    model = model1(ip_shape)

    model.fit(x_train, y_train, batch_size=256, verbose=1, epochs=30)
    model.save('crime_model.h5')

    yhats_train = model.predict(x_test, batch_size=256)
    value = np.argmax(yhats_train, axis=1)

    model_dict = {}
    model_dict['ohe_w'] = enc_w_fit
    model_dict['ohe_h'] = enc_h_fit
    model_dict['lbl_w'] = lbl_w_fit

    save_model(model_dict)
