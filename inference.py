import joblib
import os
import json
import numpy as np
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = "/opt/ml/model/model.joblib"

def load_model():
    global model
    model = joblib.load(MODEL_PATH)

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify(status="ok")

@app.route("/invocations", methods=["POST"])
def predict():
    data = request.get_json()
    input_data = np.array(data["input"])
    prediction = model.predict(input_data).tolist()
    return jsonify(prediction=prediction)

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=8080)

