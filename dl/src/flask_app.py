#!/usr/bin/env python3

from flask import Flask
from flask import request
from flask.json import jsonify
from flask_cors import CORS, cross_origin
from predict_data import predict

app = Flask(__name__)
CORS(app)

@app.route("/predict/<week>")
def pred(week):
    y_hat = predict(week).tolist()
    return jsonify({"result": y_hat})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port='8888')