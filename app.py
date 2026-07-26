"""
SolarIQ — Flask backend
Loads the trained XGBoost model (model.pkl) and serves the two endpoints
the dashboard HTML expects: /api/stats and /api/analyze.
"""

import pickle
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
import os

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    STORE = pickle.load(f)

MODEL = STORE["model"]
SCALER = STORE["scaler"]
WINDOW = STORE["window"]
DATASET_RECORDS = STORE["dataset_records"]
AVG_IRRADIANCE_NORM = STORE["avg_irradiance"]
PEAK_IRRADIANCE_NORM = STORE["peak_irradiance"]
LAST_WINDOW_SCALED = STORE["last_window_scaled"]

IRR_TO_WM2 = 1000.0

APPLIANCE_WATTS = {
    "EV Charging": 7000,
    "Washing Machine": 1500,
    "Air Conditioner": 1500,
    "Water Heater": 2000,
}

PANEL_CAPACITY_KW = 5.0
ELECTRICITY_RATE_INR = 8.0
CO2_FACTOR_KG_PER_KWH = 0.82


def forecast_next_24_hours(cloud_cover, temperature, wind_speed):
    window = LAST_WINDOW_SCALED.copy()
    preds_scaled = []
    for _ in range(24):
        x = window.reshape(1, -1)
        p = MODEL.predict(x)[0]
        preds_scaled.append(p)
        window = np.append(window[1:], p)

    preds_norm = SCALER.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).reshape(-1)
    preds_norm = np.clip(preds_norm, 0, None)

    cloud_factor = 1 - (cloud_cover / 100.0) * 0.6
    wind_factor = 1 + (wind_speed / 100.0) * 0.05
    temp_factor = 1 - max(0, (temperature - 35)) * 0.004
    adjustment = max(0.05, cloud_factor * wind_factor * temp_factor)

    return preds_norm * adjustment


@app.route("/")
def index():
    return send_from_directory(os.getcwd(), "index.html")


@app.route("/api/stats")
def stats():
    return jsonify({
        "total_records": DATASET_RECORDS,
        "avg_irradiance": round(AVG_IRRADIANCE_NORM * IRR_TO_WM2, 1),
        "peak_irradiance": round(PEAK_IRRADIANCE_NORM * IRR_TO_WM2, 1),
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    hour = int(data.get("hour", 12))
    temperature = float(data.get("temperature", 30))
    cloud_cover = float(data.get("cloud_cover", 20))
    wind_speed = float(data.get("wind_speed", 5))
    appliance = data.get("appliance", "EV Charging")

    timeline_norm = forecast_next_24_hours(cloud_cover, temperature, wind_speed)
    timeline = [{"hour": h, "irradiance": round(float(v) * IRR_TO_WM2, 1)} for h, v in enumerate(timeline_norm)]

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
