import math
from flask import Flask, jsonify, request
from flask_cors import CORS
from main import finalpredict

app = Flask(__name__)
CORS(app)

@app.route("/predict")
def predict():
    seed = request.args.get("seed", 42, type=int)
    return jsonify(finalpredict(seed))


if __name__ == "__main__":
    app.run(port=5000)
