# app.py

from flask import Flask, render_template, request
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("wine_cultivar_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            # Get input values from form
            alcohol = float(request.form["alcohol"])
            malic_acid = float(request.form["malic_acid"])
            alcalinity_of_ash = float(request.form["alcalinity_of_ash"])
            total_phenols = float(request.form["total_phenols"])
            flavanoids = float(request.form["flavanoids"])
            proline = float(request.form["proline"])

            # Create input array
            input_data = np.array([[
                alcohol,
                malic_acid,
                alcalinity_of_ash,
                total_phenols,
                flavanoids,
                proline
            ]])

            # Scale input data
            input_scaled = scaler.transform(input_data)

            # Make prediction
            pred_class = model.predict(input_scaled)[0]

            prediction = f"Cultivar {pred_class}"

        except Exception as e:
            prediction = "Invalid input. Please enter valid numerical values."

    return render_template("index.html", prediction=prediction)



import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

