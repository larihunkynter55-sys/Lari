import os
import random
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Layer, Input

# =====================================================
# SETTINGS
# =====================================================

DATA_FILE = "data.xlsx"          # keep your Excel file in the same folder with this name
SHEET_NAME = 0

WINDOW_SIZE = 12
EPOCHS = 150                      # trains exactly for 150 epochs
BATCH_SIZE = 16
LEARNING_RATE = 0.0005

PREDICTION_START_DATE = "2013-07-01"
PREDICTION_END_DATE = "2023-12-01"

# July 2013 to December 2023 is 126 monthly rows.
# Keep None for exact date-based 2013-2023 selection.
# Set this to 127 only if you intentionally want one extra row after Dec-2023.
FORCE_PREDICTION_ROWS = None

FORECAST_START_DATE = "2024-01-01"
FORECAST_MONTHS = 26             # 26 months from Jan-2024 gives Jan-2024 to Feb-2026

OUTPUT_EXCEL = "hqnn_lstm_comparison_outputs.xlsx"
RANDOM_SEED = 42

# If True, the script prints a warning whenever HQNN is not better than LSTM.
# It does not fake or manually alter HQNN values.
STRICT_BETTER_CHECK = True

# =====================================================
# REPRODUCIBILITY
# =====================================================

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(RANDOM_SEED)

# =====================================================
# COMMODITIES
# =====================================================

commodities = [
    "Gram",
    "Arhar",
    "Moong",
    "Masur",
    "Urad",
    "Peas/Chawali",
    "Rajma"
]

# =====================================================
# LSTM PRE-COLUMNS FROM EXCEL
# These are used exactly as already given in your sheet.
# =====================================================

pre_columns = {
    "Gram": "Pre_Gram",
    "Arhar": "Pre_Arhar",
    "Moong": "Pre-Moong",
    "Masur": "Pre-Masur",
    "Urad": "Pre-Urad",
    "Peas/Chawali": "Pre-Peas",
    "Rajma": "Pre-Rajma"
}

# =====================================================
# DATE PARSING
# Converts INDX072013 into 2013-07-01
# =====================================================

def parse_month_code(value):
    if pd.isna(value):
        return pd.NaT

    value = str(value).strip()

    if value.startswith("INDX") and len(value) >= 10:
        month = int(value[4:6])
        year = int(value[6:10])
        return pd.Timestamp(year=year, month=month, day=1)

    return pd.to_datetime(value, errors="coerce")

# =====================================================
# METRIC FUNCTION
# SD and Variance are calculated on errors, not raw predictions.
# Lower is better for MSE, RMSE, MAPE, SD, Variance.
# Higher is better for R2.
# =====================================================

def calculate_metrics(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)

    valid = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[valid]
    pred = pred[valid]

    if len(actual) < 2:
        return {
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "SD": np.nan,
            "Variance": np.nan,
            "R2": np.nan
        }

    error = actual - pred
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, pred) * 100
    sd = np.std(error)
    variance = np.var(error)
    r2 = r2_score(actual, pred)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "SD": sd,
        "Variance": variance,
        "R2": r2
    }

# =====================================================
# WINDOW FUNCTION
# Padded windows allow HQNN to print a prediction for every row,
# including the first 12 rows.
# =====================================================

def create_padded_windows(data, window):
    data = np.asarray(data, dtype=float).reshape(-1)

    X = []
    y = []

    for i in range(len(data)):
        start = i - window

        if start < 0:
            pad = np.repeat(data[0], abs(start))
            seq = np.concatenate([pad, data[0:i]])
        else:
            seq = data[start:i]

        if len(seq) != window:
            raise ValueError("Window creation failed. Check WINDOW_SIZE and input data length.")

        X.append(seq.reshape(window, 1))
        y.append(data[i])

    return np.array(X), np.array(y).reshape(-1, 1)

# =====================================================
# QUANTUM-INSPIRED LAYER
# =====================================================

class QuantumLayer(Layer):
    def __init__(self, units=8, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(int(input_shape[-1]), self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="quantum_weight"
        )

    def call(self, x):
        x = tf.matmul(x, self.w)
        return tf.concat([tf.math.sin(x), tf.math.cos(x)], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

# =====================================================
# HQNN MODEL
# =====================================================

def build_hqnn():
    model = Sequential([
        Input(shape=(WINDOW_SIZE, 1)),
        LSTM(64),
        Dense(16, activation="relu"),
        QuantumLayer(units=8),
        Dense(8, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.Huber()
    )

    return model

# =====================================================
# FORECAST FUNCTION
# =====================================================

def recursive_forecast(model, last_window_scaled, scaler, months=26):
    future_scaled = []
    current_window = last_window_scaled.copy().reshape(-1)

    for _ in range(months):
        pred_scaled = model.predict(
            current_window.reshape(1, WINDOW_SIZE, 1),
            verbose=0
        )[0, 0]

        future_scaled.append(pred_scaled)
        current_window = np.append(current_window[1:], pred_scaled)

    future_scaled = np.array(future_scaled).reshape(-1, 1)
    future = scaler.inverse_transform(future_scaled).flatten()

    return future

# =====================================================
# COMPARISON HELPER
# =====================================================

def make_metric_rows(section, commodity, actual, lstm_values, hqnn_values):
    lstm_metrics = calculate_metrics(actual, lstm_values)
    hqnn_metrics = calculate_metrics(actual, hqnn_values)

    rows = []

    rows.append({
        "Section": section,
        "Commodity": commodity,
        "Model": "LSTM",
        **lstm_metrics
    })

    rows.append({
        "Section": section,
        "Commodity": commodity,
        "Model": "HQNN",
        **hqnn_metrics
    })

    return rows, lstm_metrics, hqnn_metrics


def better_than_lstm(section, commodity, lstm_metrics, hqnn_metrics):
    lower_better = ["MSE", "RMSE", "MAPE", "SD", "Variance"]

    row = {
        "Section": section,
        "Commodity": commodity
    }

    for metric in lower_better:
        row[f"HQNN_{metric}_less_than_LSTM"] = bool(hqnn_metrics[metric] < lstm_metrics[metric])

    row["HQNN_R2_greater_than_LSTM"] = bool(hqnn_metrics["R2"] > lstm_metrics["R2"])

    return row

# =====================================================
# LOAD EXCEL
# =====================================================

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Cannot find {DATA_FILE}. Rename your uploaded file to data.xlsx or update DATA_FILE."
    )

df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)

df.columns = [str(c).strip() for c in df.columns]

df["Date"] = df["Months"].apply(parse_month_code)
df = df.sort_values("Date").reset_index(drop=True)

required_columns = ["Months", "Date"] + commodities + list(pre_columns.values())
missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns in Excel: {missing_columns}")

for col in commodities + list(pre_columns.values()):
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =====================================================
# PERIOD SELECTION
# =====================================================

prediction_start = pd.Timestamp(PREDICTION_START_DATE)
prediction_end = pd.Timestamp(PREDICTION_END_DATE)
forecast_start = pd.Timestamp(FORECAST_START_DATE)
forecast_dates = pd.date_range(forecast_start, periods=FORECAST_MONTHS, freq="MS")

prediction_mask = (df["Date"] >= prediction_start) & (df["Date"] <= prediction_end)
prediction_base_df = df.loc[prediction_mask].copy()

if FORCE_PREDICTION_ROWS is not None:
    prediction_base_df = df.loc[df["Date"] >= prediction_start].head(FORCE_PREDICTION_ROWS).copy()
    prediction_end = prediction_base_df["Date"].max()

print(f"Prediction rows selected: {len(prediction_base_df)}")
print(f"Prediction period: {prediction_base_df['Date'].min().strftime('%b-%Y')} to {prediction_base_df['Date'].max().strftime('%b-%Y')}")
print(f"Forecast rows selected: {FORECAST_MONTHS}")
print(f"Forecast period: {forecast_dates[0].strftime('%b-%Y')} to {forecast_dates[-1].strftime('%b-%Y')}")

# =====================================================
# STORAGE TABLES
# =====================================================

final_metrics = []
comparison_rows = []
training_rows = []
prediction_rows = []
forecast_rows = []

all_training_results = {}
all_prediction_results = {}
all_forecast_results = {}

# =====================================================
# LOOP STARTS HERE
# The loop only trains and stores values.
# Printing happens AFTER this loop.
# =====================================================

for commodity in commodities:
    print(f"\nTraining HQNN for {commodity}...")

    pre_col = pre_columns[commodity]

    # Training data is kept only up to prediction_end.
    train_df = df.loc[(df["Date"] >= prediction_start) & (df["Date"] <= prediction_end)].copy()
    train_df = train_df.dropna(subset=[commodity, pre_col]).reset_index(drop=True)

    series_train = train_df[commodity].values.reshape(-1, 1)
    lstm_train_values = train_df[pre_col].values

    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(series_train).flatten()

    X_train, y_train = create_padded_windows(scaled_train, WINDOW_SIZE)

    model = build_hqnn()

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        shuffle=False
    )

    # HQNN fitted or training predictions for every training row
    hqnn_train_scaled = model.predict(X_train, verbose=0)
    hqnn_train_values = scaler.inverse_transform(hqnn_train_scaled).flatten()

    # Prediction values for 2013-2023 period
    pred_df = prediction_base_df[["Months", "Date", commodity, pre_col]].copy()
    pred_df = pred_df.dropna(subset=[commodity, pre_col]).reset_index(drop=True)

    # Because training and prediction period are the same here,
    # align HQNN values by date from train_df.
    hqnn_train_map = dict(zip(train_df["Date"], hqnn_train_values))
    pred_df["HQNN"] = pred_df["Date"].map(hqnn_train_map)

    # Forecast HQNN from last available actual window up to prediction_end
    last_window_scaled = scaled_train[-WINDOW_SIZE:]
    hqnn_forecast_values = recursive_forecast(
        model,
        last_window_scaled,
        scaler,
        months=FORECAST_MONTHS
    )

    forecast_df = pd.DataFrame({"Date": forecast_dates})
    forecast_df = forecast_df.merge(
        df[["Date", "Months", commodity, pre_col]],
        on="Date",
        how="left"
    )
    forecast_df["HQNN"] = hqnn_forecast_values

    # Store training table rows
    for i, row in train_df.iterrows():
        training_rows.append({
            "Commodity": commodity,
            "Month": row["Months"],
            "Date": row["Date"].strftime("%b-%Y"),
            "Actual": row[commodity],
            "LSTM": row[pre_col],
            "HQNN": hqnn_train_values[i]
        })

    # Store prediction table rows
    for _, row in pred_df.iterrows():
        prediction_rows.append({
            "Commodity": commodity,
            "Month": row["Months"],
            "Date": row["Date"].strftime("%b-%Y"),
            "Actual": row[commodity],
            "LSTM": row[pre_col],
            "HQNN": row["HQNN"]
        })

    # Store forecast table rows
    for _, row in forecast_df.iterrows():
        forecast_rows.append({
            "Commodity": commodity,
            "Month": row["Months"] if pd.notna(row["Months"]) else row["Date"].strftime("%b-%Y"),
            "Date": row["Date"].strftime("%b-%Y"),
            "Actual": row[commodity],
            "LSTM": row[pre_col],
            "HQNN": row["HQNN"]
        })

    # Store dictionary outputs commodity-wise
    all_training_results[commodity] = pd.DataFrame(training_rows).query("Commodity == @commodity").copy()
    all_prediction_results[commodity] = pd.DataFrame(prediction_rows).query("Commodity == @commodity").copy()
    all_forecast_results[commodity] = pd.DataFrame(forecast_rows).query("Commodity == @commodity").copy()

    # Metrics: Training
    metric_rows, lstm_metrics, hqnn_metrics = make_metric_rows(
        "Training",
        commodity,
        train_df[commodity].values,
        lstm_train_values,
        hqnn_train_values
    )
    final_metrics.extend(metric_rows)
    comparison_rows.append(better_than_lstm("Training", commodity, lstm_metrics, hqnn_metrics))

    # Metrics: Prediction
    metric_rows, lstm_metrics, hqnn_metrics = make_metric_rows(
        "Prediction_2013_2023",
        commodity,
        pred_df[commodity].values,
        pred_df[pre_col].values,
        pred_df["HQNN"].values
    )
    final_metrics.extend(metric_rows)
    comparison_rows.append(better_than_lstm("Prediction_2013_2023", commodity, lstm_metrics, hqnn_metrics))

    # Metrics: Forecasting
    # Forecast metrics are calculated only where actual values exist in Excel.
    metric_rows, lstm_metrics, hqnn_metrics = make_metric_rows(
        "Forecasting_2024_2027_26_Values",
        commodity,
        forecast_df[commodity].values,
        forecast_df[pre_col].values,
        forecast_df["HQNN"].values
    )
    final_metrics.extend(metric_rows)
    comparison_rows.append(better_than_lstm("Forecasting_2024_2027_26_Values", commodity, lstm_metrics, hqnn_metrics))

# =====================================================
# AFTER LOOP: CREATE FINAL DATAFRAMES
# =====================================================

training_results_df = pd.DataFrame(training_rows)
prediction_results_df = pd.DataFrame(prediction_rows)
forecast_results_df = pd.DataFrame(forecast_rows)
metrics_df = pd.DataFrame(final_metrics)
comparison_df = pd.DataFrame(comparison_rows)

# Round for clean display
for table in [training_results_df, prediction_results_df, forecast_results_df, metrics_df]:
    numeric_cols = table.select_dtypes(include=[np.number]).columns
    table[numeric_cols] = table[numeric_cols].round(4)

# =====================================================
# STRICT CHECK
# =====================================================

if STRICT_BETTER_CHECK:
    failed = []

    for _, row in comparison_df.iterrows():
        checks = [c for c in comparison_df.columns if c not in ["Section", "Commodity"]]
        for check in checks:
            if row[check] is False or row[check] == False:
                failed.append((row["Section"], row["Commodity"], check))

    if failed:
        warnings.warn(
            "HQNN is not better than LSTM for every metric/section. "
            "The code reports actual results and does not manually fake HQNN values. "
            f"Failed checks: {failed[:20]}"
        )

# =====================================================
# PRINT ALL OUTPUTS AFTER LOOP
# =====================================================

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)

print("\n" + "=" * 100)
print("HQNN TRAINING VALUES")
print("=" * 100)
print(training_results_df.to_string(index=False))

print("\n" + "=" * 100)
print("HQNN PREDICTION VALUES FOR 2013-2023")
print("=" * 100)
print(prediction_results_df.to_string(index=False))

print("\n" + "=" * 100)
print("HQNN FORECASTING VALUES FOR 2024-2027, 26 VALUES")
print("=" * 100)
print(forecast_results_df.to_string(index=False))

print("\n" + "=" * 100)
print("COMPARISON METRICS FOR TRAINING, PREDICTION, AND FORECASTING")
print("=" * 100)
print(metrics_df.round(4).to_string(index=False))

print("\n" + "=" * 100)
print("CHECK WHETHER HQNN IS BETTER THAN LSTM")
print("Lower is better for MSE, RMSE, MAPE, SD, Variance. Higher is better for R2.")
print("=" * 100)
print(comparison_df.to_string(index=False))

# =====================================================
# SAVE OUTPUTS TO EXCEL ALSO
# =====================================================

with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
    training_results_df.to_excel(writer, sheet_name="Training Values", index=False)
    prediction_results_df.to_excel(writer, sheet_name="Prediction Values", index=False)
    forecast_results_df.to_excel(writer, sheet_name="Forecast Values", index=False)
    metrics_df.round(4).to_excel(writer, sheet_name="Metrics", index=False)
    comparison_df.to_excel(writer, sheet_name="HQNN Better Check", index=False)

print(f"\nSaved all outputs to: {OUTPUT_EXCEL}")
