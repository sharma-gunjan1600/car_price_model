from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open('car_price_model.pkl', 'rb'))

# Load cleaned data
car = pd.read_csv("clean_car.csv")


@app.route('/')
def index():
    return render_template(
        "index.html",
        names=sorted(car['name'].unique()),
        years=sorted(car['year'].unique()),
        fuels=sorted(car['fuel_type'].unique()),
        selected_name=None,
        selected_year=None,
        selected_fuel=None,
        selected_company=None,
        selected_kms="",
        label_text=None,
        price_range_only=None
    )


@app.route('/get_details', methods=['POST'])
def get_details():
    car_name = request.form['car_name']
    filtered = car[car['name'] == car_name]

    kms_min = int(filtered['kms_driven'].min())
    kms_max = int(filtered['kms_driven'].max())

    # Expand range if identical
    if kms_min == kms_max:
        buffer = int(kms_min * 0.10)
        kms_min -= buffer
        kms_max += buffer

    return jsonify({
        "company": filtered['company'].iloc[0],
        "kms_min": kms_min,
        "kms_max": kms_max
    })


@app.route('/predict', methods=['POST'])
def predict():

    name = request.form['name']
    year = int(request.form['year'])
    fuel = request.form['fuel']
    company = request.form['company']
    kms_input = request.form['kms']

    # KM required
    if kms_input.strip() == "":
        return render_template(
            "index.html",
            label_text=None,
            price_range_only=None,
            prediction_text="❌ Please enter Kilometers Driven.",
            names=sorted(car['name'].unique()),
            years=sorted(car['year'].unique()),
            fuels=sorted(car['fuel_type'].unique()),
            selected_name=name,
            selected_year=year,
            selected_fuel=fuel,
            selected_company=company,
            selected_kms=""
        )

    kms = int(kms_input)

    # Prepare input for model
    input_df = pd.DataFrame(
        [[name, company, year, kms, fuel]],
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
    )

    pred = model.predict(input_df)[0]
    pred = max(pred, 50000)

    # Price Range ±10%
    lower = int(pred * 0.90)
    upper = int(pred * 1.10)

    label_text = "Estimated Price Range:"
    price_range_only = f"₹{lower:,} - ₹{upper:,}"

    return render_template(
        "index.html",
        label_text=label_text,
        price_range_only=price_range_only,
        names=sorted(car['name'].unique()),
        years=sorted(car['year'].unique()),
        fuels=sorted(car['fuel_type'].unique()),
        selected_name=name,
        selected_year=year,
        selected_fuel=fuel,
        selected_company=company,
        selected_kms=kms
    )


if __name__ == "__main__":
    app.run(debug=True)