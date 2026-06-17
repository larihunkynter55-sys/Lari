import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# LOAD EXCEL
# =====================================================
df = pd.read_excel("data")

# =====================================================
# COLUMNS
# =====================================================

commodities = {

    "Gram":"Pre_Gram",

    "Arhar":"Pre_Arhar",

    "Moong":"Pre-Moong",

    "Masur":"Pre-Masur",

    "Urad":"Pre-Urad",

    "Peas/Chawali":"Pre-Peas",

    "Rajma":"Pre-Rajma"

}

# =====================================================
# PLOTS
# =====================================================

for commodity, pre_col in commodities.items():

    safe_name = (
        commodity
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )

    actual = df[commodity].dropna().values

    lstm = df[pre_col].dropna().values

    n = min(
        len(actual),
        len(lstm)
    )

    actual = actual[:n]
    lstm = lstm[:n]

    # =================================================
    # TRAINING PLOT
    # =================================================

    plt.figure(figsize=(14,6))

    plt.plot(
        actual,
        label="Actual",
        linewidth=2
    )

    plt.plot(
        lstm,
        label="LSTM",
        linewidth=2
    )

    plt.title(
        f"{commodity} Training Comparison"
    )

    plt.xlabel("Months")

    plt.ylabel("Price")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"{safe_name}_Training.png",
        dpi=300
    )

    plt.close()

    # =================================================
    # PREDICTION PLOT
    # =================================================

    plt.figure(figsize=(14,6))

    plt.plot(
        actual,
        label="Actual",
        linewidth=2
    )

    plt.plot(
        lstm,
        label="LSTM Prediction",
        linewidth=2
    )

    plt.title(
        f"{commodity} Prediction Comparison"
    )

    plt.xlabel("Months")

    plt.ylabel("Price")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"{safe_name}_Prediction.png",
        dpi=300
    )

    plt.close()

    # =================================================
    # FORECAST PLOT
    # LAST 26 VALUES
    # =================================================

    forecast_lstm = lstm[-26:]

    forecast_months = np.arange(
        1,
        27
    )

    plt.figure(figsize=(14,6))

    plt.plot(
        forecast_months,
        forecast_lstm,
        marker="o",
        linewidth=2,
        label="LSTM Forecast"
    )

    plt.title(
        f"{commodity} Forecast 2024-2026"
    )

    plt.xlabel("Forecast Month")

    plt.ylabel("Price")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"{safe_name}_Forecast.png",
        dpi=300
    )

    plt.close()

    # =================================================
    # FINAL COMBINED PLOT
    # =================================================

    plt.figure(figsize=(18,8))

    plt.plot(
        actual,
        label="Actual",
        linewidth=2
    )

    plt.plot(
        lstm,
        label="LSTM",
        linewidth=2
    )

    forecast_x = np.arange(
        len(actual)-26,
        len(actual)
    )

    plt.plot(
        forecast_x,
        forecast_lstm,
        "--",
        linewidth=3,
        label="Forecast"
    )

    plt.title(
        f"{commodity} Complete Series"
    )

    plt.xlabel("Months")

    plt.ylabel("Price")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"{safe_name}_Final.png",
        dpi=300
    )

    plt.show()

print("All plots generated successfully.")