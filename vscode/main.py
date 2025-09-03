from flask import Flask, jsonify, request
import joblib
import pandas as pd

from flask_cors import CORS   # <-- add this

app = Flask(__name__)
CORS(app)  # <-- enable CORS so HTML form can call API

# Load model and pipeline
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert JSON to DataFrame
        df = pd.DataFrame([data])

        # Transform input using pipeline
        processed_data = pipeline.transform(df)

        # Predict
        prediction = model.predict(processed_data)[0]

        return jsonify({"Predicted Price": float(prediction)})
    
    except Exception as e:
        return jsonify({"Error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
