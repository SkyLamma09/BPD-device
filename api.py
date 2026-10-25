from flask import Flask, jsonify
from flask_cors import CORS
from combination import finalpredict

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["GET"])
def predict():
    try:
        result = finalpredict()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)
