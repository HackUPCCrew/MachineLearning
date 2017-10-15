#!/usr/bin/env python3

from flask import Flask
from flask import request
from flask.json import jsonify
from flask_cors import CORS, cross_origin
from predict_data import predict
import pickle
import keras
from keras.models import Sequential
from keras.layers.core import Dense

app = Flask(__name__)
CORS(app)


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


@app.route("/predict/<week>")
def pred(week):
    y_hat = predict(week, model_dict, model).tolist()
    return jsonify({"result": y_hat})


if __name__ == "__main__":
    LOAD_PICKLE = '/Users/krishnakalyan3/MOOC/MachineLearning/dl/src/model_trans_v1.pkl'
    LOAD_MODEL = '/Users/krishnakalyan3/MOOC/MachineLearning/dl/src/crime_model.h5'

    # Load Models
    model = model1((24, 31))
    model.load_weights(LOAD_MODEL)
    model_dict = load_pickle()
    app.run(host='0.0.0.0', port='8888')