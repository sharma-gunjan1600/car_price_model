from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

# ------------------------------
# SAFE FILE LOADING (Render compatible)
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def safe_load(path):
    return os.path.join(BASE_DIR, path)


# Load model
model = pickle.load(open(safe_load("car_price_model.pkl"), "rb"))

# Load cleaned car dataset
car = pd.read_csv(safe_load("clean_car.csv"))


# ------------------------------
# ROUTES
# ------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        names=sorted(car['name'].unique()),
        years=list(range(1995, 2025 + 1)),   # ✅ FIXED YEAR RANGE
        fuels=sorted(car['fuel_type'].unique()),
        selected_name=None,
        selected_year=None,
        selected_fuel=None,
        selected_company=None,
        selected_kms="",
        label_text=None,
        price_range_only=None
    )


# ------------------------------
# GET DETAILS API
# ------------------------------
@app.route("/get_details", methods=["POST"])
def get_details():
    car_name = request.form["car_name"]

    filtered = car[car["name"] == car_name]

    # Prevent crashes
    if filtered.empty:
        return jsonify({
            "company": "",
            "kms_min": 1000,
            "kms_max": 5000
        })

    kms_min = int(filtered["kms_driven"].min())
    kms_max = int(filtered["kms_driven"].max())

    # If all values same → expand by 10%
    if kms_min == kms_max:
        buffer = int(kms_min * 0.10)
        kms_min -= buffer
        kms_max += buffer

    return jsonify({
        "company": filtered["company"].iloc[0],
        "kms_min": kms_min,
        "kms_max": kms_max
    })


# ------------------------------
# PREDICT PRICE
# ------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    name = request.form["name"]
    year = int(request.form["year"])
    fuel = request.form["fuel"]
    company = request.form["company"]
    kms_input = request.form["kms"]

    # Prevent crash on empty km
    if kms_input.strip() == "":
        return render_template(
            "index.html",
            label_text=None,
            price_range_only=None,
            prediction_text="❌ Please enter Kilometers Driven.",
            names=sorted(car['name'].unique()),
            years=list(range(1995, 2025 + 1)),   # ✅ FIXED YEAR RANGE
            fuels=sorted(car['fuel_type'].unique()),
            selected_name=name,
            selected_year=year,
            selected_fuel=fuel,
            selected_company=company,
            selected_kms=""
        )

    kms = int(kms_input)

    # Construct dataframe in correct order
    input_df = pd.DataFrame(
        [[name, company, fuel, year, kms]],
        columns=["name", "company", "fuel_type", "year", "kms_driven"]
    )

    pred = model.predict(input_df)[0]
    pred = max(pred, 50000)  # Minimum price safeguard

    lower = int(pred * 0.90)
    upper = int(pred * 1.10)

    return render_template(
        "index.html",
        label_text="Estimated Price Range:",
        price_range_only=f"₹{lower:,} - ₹{upper:,}",
        names=sorted(car['name'].unique()),
        years=list(range(1995, 2025 + 1)),  # ✅ FIXED YEAR RANGE
        fuels=sorted(car['fuel_type'].unique()),
        selected_name=name,
        selected_year=year,
        selected_fuel=fuel,
        selected_company=company,
        selected_kms=kms
    )


if __name__ == "__main__":
    app.run(debug=True)