"""
SolarIQ — Flask backend
Loads the trained XGBoost model (model.pkl) and serves the two endpoints
the dashboard HTML expects: /api/stats and /api/analyze.

IMPORTANT — honest note about how prediction works:
Your 5 models were trained as TIME-SERIES forecasters (they predict the
next hour's irradiance from the previous 24 hours of irradiance values).
They do NOT take temperature/humidity/cloud_cover/wind_speed as direct
model inputs — your research paper's models never used those either.

So here's what this backend actually does:
  1. Uses the trained XGBoost model to auto-regressively forecast the
     next 24 hours of irradiance, starting from the last known window
     in your training data (this part IS real model output).
  2. Applies a simple, clearly-labeled physical adjustment using the
     cloud_cover/temperature/wind_speed form inputs (e.g. more cloud =
     less sunlight reaches the panel). This part is a reasonable
     engineering rule, not a trained ML prediction — it's what lets the
     dashboard react to the form fields at all, since the real models
     are univariate.
  3. Appliance scheduling logic (best time to run appliance, savings,
     CO2) is straightforward arithmetic on top of the forecast.
This is a common, honest pattern for turning a research forecasting
model into a demo app — worth explaining this exact split in your
README / viva if asked "does the model use temperature as input?".
"""

import pickle
import numpy as np
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    STORE = pickle.load(f)

MODEL = STORE["model"]
SCALER = STORE["scaler"]
WINDOW = STORE["window"]
RESULTS_TABLE = STORE["results_table"]
DATASET_RECORDS = STORE["dataset_records"]
AVG_IRRADIANCE_NORM = STORE["avg_irradiance"]   # 0-1 normalized scale
PEAK_IRRADIANCE_NORM = STORE["peak_irradiance"] # 0-1 normalized scale
LAST_WINDOW_SCALED = STORE["last_window_scaled"]  # last known 24 scaled values

# Convert normalized (0-1) irradiance to a display-friendly W/m^2 scale
IRR_TO_WM2 = 1000.0

# Simple appliance power ratings (Watts) for the scheduling logic
APPLIANCE_WATTS = {
    "EV Charging": 7000,
    "Washing Machine": 1500,
    "Air Conditioner": 1500,
    "Water Heater": 2000,
}

PANEL_CAPACITY_KW = 5.0        # assumed rooftop solar panel size
ELECTRICITY_RATE_INR = 8.0     # INR per kWh (approx residential rate)
CO2_FACTOR_KG_PER_KWH = 0.82   # India grid emission factor


def forecast_next_24_hours(cloud_cover, temperature, wind_speed):
    """Auto-regressively forecast the next 24 hourly irradiance values
    using the trained XGBoost model, then apply a simple weather
    adjustment factor based on the form inputs."""
    window = LAST_WINDOW_SCALED.copy()
    preds_scaled = []
    for _ in range(24):
        x = window.reshape(1, -1)
        p = MODEL.predict(x)[0]
        preds_scaled.append(p)
        window = np.append(window[1:], p)

    preds_norm = SCALER.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).reshape(-1)
    preds_norm = np.clip(preds_norm, 0, None)

    # Weather adjustment (rule-based, not model-based — see module docstring)
    cloud_factor = 1 - (cloud_cover / 100.0) * 0.6
    wind_factor = 1 + (wind_speed / 100.0) * 0.05          # light cooling/cleaning effect
    temp_factor = 1 - max(0, (temperature - 35)) * 0.004    # mild derate above 35C
    adjustment = max(0.05, cloud_factor * wind_factor * temp_factor)

    adjusted = preds_norm * adjustment
    return adjusted  # still 0-1 normalized scale, 24 values for hours 0-23


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats")
def stats():
    best_row = min(RESULTS_TABLE, key=lambda r: r["MAE"])
    return jsonify({
        "total_records": DATASET_RECORDS,
        "model_r2": round(best_row["R2"], 3),
        "model_mae": round(best_row["MAE"], 4),
        "avg_irradiance": round(AVG_IRRADIANCE_NORM * IRR_TO_WM2, 1),
        "peak_irradiance": round(PEAK_IRRADIANCE_NORM * IRR_TO_WM2, 1),
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    hour = int(data.get("hour", 12))
    temperature = float(data.get("temperature", 30))
    humidity = float(data.get("humidity", 50))
    cloud_cover = float(data.get("cloud_cover", 20))
    wind_speed = float(data.get("wind_speed", 5))
    appliance = data.get("appliance", "EV Charging")

    timeline_norm = forecast_next_24_hours(cloud_cover, temperature, wind_speed)
    timeline = [
        {"hour": h, "irradiance": round(float(v) * IRR_TO_WM2, 1)}
        for h, v in enumerate(timeline_norm)
    ]

    best_hour = int(np.argmax(timeline_norm))
    current_irradiance_wm2 = timeline[hour]["irradiance"]

    efficiency = 0.0
    if PEAK_IRRADIANCE_NORM > 0:
        efficiency = min(100.0, float(timeline_norm[hour] / PEAK_IRRADIANCE_NORM) * 100)

    if efficiency >= 70:
        status, status_msg = "PEAK", "Solar generation is at its best right now."
    elif efficiency >= 35:
        status, status_msg = "MODERATE", "Solar generation is moderate right now."
    else:
        status, status_msg = "LOW", "Solar generation is low right now."

    panel_kwh = round(float(timeline_norm[hour]) * PANEL_CAPACITY_KW, 2)
    appliance_watt = APPLIANCE_WATTS.get(appliance, 1500)
    appliance_kwh = appliance_watt / 1000.0
    net_kwh = round(panel_kwh - appliance_kwh, 2)

    used_from_solar = min(panel_kwh, appliance_kwh)
    daily_savings = round(used_from_solar * ELECTRICITY_RATE_INR, 2)
    monthly_savings = round(daily_savings * 30, 2)
    co2_saved = round(panel_kwh * CO2_FACTOR_KG_PER_KWH, 2)

    return jsonify({
        "status": status,
        "status_msg": status_msg,
        "best_hour": best_hour,
        "co2_saved": co2_saved,
        "daily_savings": daily_savings,
        "monthly_savings": monthly_savings,
        "appliance_watt": appliance_watt,
        "timeline": timeline,
        "current": {
            "irradiance": current_irradiance_wm2,
            "efficiency": round(efficiency, 1),
            "panel_kwh": panel_kwh,
            "net_kwh": net_kwh,
        },
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
