# -*- coding: utf-8 -*-
"""
SolarIQ — Model Training Script
Cleaned and fixed version of the original Colab research notebook.

Trains SARIMA, Prophet, LSTM, GRU, and XGBoost on:
  - data/solar_pune.csv  (solar irradiance, Pune, India)
  - data/AEP_hourely_data.csv (electricity demand, AEP, USA)

Fixes applied vs. the original Colab code:
  1. solar_pune.csv had no time-of-day column (only repeated dates).
     Reconstructed realistic timestamps by evenly spacing each day's
     readings across 24 hours (rows are in original chronological order).
  2. Fixed filename mismatch ('solar pune.csv' -> 'solar_pune.csv').
  3. Reordered code so it runs top-to-bottom without NameErrors.
  4. Saves the best-performing model (XGBoost) to model.pkl for the
     Flask app to load.
"""

import os
import warnings
import pickle
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
np.random.seed(42)

DATA_DIR = "data"
SOLAR_CSV = os.path.join(DATA_DIR, "solar_pune.csv")
AEP_CSV = os.path.join(DATA_DIR, "AEP_hourely_data.csv")


# ---------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------
def evaluate_series(y_true, y_pred):
    y_true = np.array(y_true).astype(float).reshape(-1)
    y_pred = np.array(y_pred).astype(float).reshape(-1)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    nz = y_true != 0
    mape = float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100) if nz.sum() else np.nan
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, mape, r2


# ---------------------------------------------------------------------
# 1) Load + fix solar data (reconstruct timestamps)
# ---------------------------------------------------------------------
def load_solar(path=SOLAR_CSV):
    df = pd.read_csv(path)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], format="%d-%m-%Y")

    # Reconstruct time-of-day: within each date, evenly space rows
    # across 24 hours (original 15-min interval order preserved).
    df["row_in_day"] = df.groupby("DATE_TIME").cumcount()
    df["count_in_day"] = df.groupby("DATE_TIME")["DATE_TIME"].transform("count")
    minutes_offset = (df["row_in_day"] / df["count_in_day"]) * 24 * 60
    df["DATE_TIME"] = df["DATE_TIME"] + pd.to_timedelta(minutes_offset, unit="m")

    df = df.sort_values("DATE_TIME").set_index("DATE_TIME")
    return df[["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"]]


def load_aep(path=AEP_CSV):
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%d-%m-%Y", errors="coerce")
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime").set_index("Datetime")
    return df[["AEP_MW"]]


def preprocess_series(df, target_col, rule="h", agg="mean"):
    s = df[[target_col]].copy()
    s = s.resample(rule).mean() if agg == "mean" else s.resample(rule).sum()
    s = s.interpolate(method="time").ffill().bfill()
    return s


def time_split(ts, train_frac=0.8):
    n = len(ts)
    return ts.iloc[: int(n * train_frac)], ts.iloc[int(n * train_frac):]


# ---------------------------------------------------------------------
# 2) Models
# ---------------------------------------------------------------------
def run_sarima(train, test, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24)):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    preds = res.forecast(len(test))
    return preds.values


def run_prophet(train_series, test_series):
    df_train = train_series.reset_index()
    df_train.columns = ["ds", "y"]
    m = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=True)
    m.fit(df_train)
    future = m.make_future_dataframe(periods=len(test_series), freq="h")
    forecast = m.predict(future)
    return forecast.set_index("ds")["yhat"].iloc[-len(test_series):].values


def create_sequences(arr, window):
    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i - window:i])
        y.append(arr[i])
    return np.array(X), np.array(y)


def run_rnn(series_values, model_type="LSTM", window=24, epochs=15):
    arr = series_values.reshape(-1, 1).astype("float32")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(arr)
    X, y = create_sequences(scaled, window)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    X_train = X_train.reshape((-1, window, 1))
    X_test = X_test.reshape((-1, window, 1))

    model = Sequential()
    layer = LSTM if model_type == "LSTM" else GRU
    model.add(layer(50, activation="tanh", input_shape=(window, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, callbacks=[es])

    preds_scaled = model.predict(X_test, verbose=0)
    preds = scaler.inverse_transform(preds_scaled).reshape(-1)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    return preds, y_true


def run_xgboost(series_values, window=24):
    arr = series_values.reshape(-1, 1).astype("float32")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(arr)
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window:i].flatten())
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200, max_depth=4)
    model.fit(X_train, y_train)

    preds_scaled = model.predict(X_test)
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    return preds, y_true, model, scaler, window


# ---------------------------------------------------------------------
# 3) Run everything on the SOLAR dataset (this is what powers the app)
# ---------------------------------------------------------------------
def main():
    print("Loading solar data...")
    solar_df = load_solar()
    solar_ts = preprocess_series(solar_df, "IRRADIATION", rule="h")
    print(f"Solar hourly series length: {len(solar_ts)}")

    solar_train, solar_test = time_split(solar_ts)

    results = {"Model": [], "MAE": [], "RMSE": [], "MAPE": [], "R2": []}

    print("Running SARIMA...")
    sarima_preds = run_sarima(solar_train["IRRADIATION"], solar_test["IRRADIATION"])
    m = evaluate_series(solar_test["IRRADIATION"].values, sarima_preds)
    results["Model"].append("SARIMA")
    for k, v in zip(["MAE", "RMSE", "MAPE", "R2"], m):
        results[k].append(v)
    print("SARIMA:", m)

    print("Running Prophet...")
    prophet_preds = run_prophet(solar_train["IRRADIATION"], solar_test["IRRADIATION"])
    m = evaluate_series(solar_test["IRRADIATION"].values, prophet_preds)
    results["Model"].append("Prophet")
    for k, v in zip(["MAE", "RMSE", "MAPE", "R2"], m):
        results[k].append(v)
    print("Prophet:", m)

    print("Running LSTM...")
    lstm_preds, lstm_true = run_rnn(solar_ts["IRRADIATION"].values, "LSTM")
    m = evaluate_series(lstm_true, lstm_preds)
    results["Model"].append("LSTM")
    for k, v in zip(["MAE", "RMSE", "MAPE", "R2"], m):
        results[k].append(v)
    print("LSTM:", m)

    print("Running GRU...")
    gru_preds, gru_true = run_rnn(solar_ts["IRRADIATION"].values, "GRU")
    m = evaluate_series(gru_true, gru_preds)
    results["Model"].append("GRU")
    for k, v in zip(["MAE", "RMSE", "MAPE", "R2"], m):
        results[k].append(v)
    print("GRU:", m)

    print("Running XGBoost...")
    xgb_preds, xgb_true, xgb_model, xgb_scaler, window = run_xgboost(solar_ts["IRRADIATION"].values)
    m = evaluate_series(xgb_true, xgb_preds)
    results["Model"].append("XGBoost")
    for k, v in zip(["MAE", "RMSE", "MAPE", "R2"], m):
        results[k].append(v)
    print("XGBoost:", m)

    results_df = pd.DataFrame(results)
    print("\n=== FINAL RESULTS (Solar irradiance, hourly) ===")
    print(results_df.to_string(index=False))
    results_df.to_csv("model_results.csv", index=False)

    # Save last known window (scaled) so the Flask app can forecast forward
    full_scaled = xgb_scaler.transform(solar_ts["IRRADIATION"].values.reshape(-1, 1)).reshape(-1)
    last_window_scaled = full_scaled[-window:]

    # Save the best model (XGBoost, per paper's finding) for the Flask app
    with open("model.pkl", "wb") as f:
        pickle.dump({
            "model": xgb_model,
            "scaler": xgb_scaler,
            "window": window,
            "results_table": results_df.to_dict(orient="records"),
            "dataset_records": len(solar_ts),
            "avg_irradiance": float(solar_ts["IRRADIATION"].mean()),
            "peak_irradiance": float(solar_ts["IRRADIATION"].max()),
            "last_window_scaled": last_window_scaled,
        }, f)
    print("\nSaved model.pkl")


if __name__ == "__main__":
    main()
