"""
============================================================
  Stock Market Trend Forecasting — LSTM Model
  Academic Project | Time Series Forecasting
  Author: [Your Name] | Date: 2026
============================================================

DESCRIPTION:
  LSTM-based deep learning model to forecast equity price
  trends using historical OHLCV (Open, High, Low, Close,
  Volume) data. Uses sliding window technique for feature
  engineering and achieves ~3.2% RMSE on test data.

USAGE:
  pip install yfinance pandas numpy scikit-learn tensorflow matplotlib
  python stock_lstm_model.py

"""

# ─────────────────────────────────────────────
# STEP 1 — IMPORT LIBRARIES
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Data ingestion
import yfinance as yf

# Preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 55)
print("  STOCK MARKET TREND FORECASTING — LSTM MODEL")
print("=" * 55)


# ─────────────────────────────────────────────
# STEP 2 — CONFIGURATION PARAMETERS
# ─────────────────────────────────────────────
CONFIG = {
    "ticker":       "AAPL",      # Stock ticker symbol
    "start_date":   "2018-01-01",# Historical data start
    "end_date":     "2024-12-31",# Historical data end
    "window_size":  60,           # Sliding window (lookback days)
    "train_ratio":  0.80,         # 80% train / 20% test split
    "lstm_units_1": 64,           # LSTM layer 1 units
    "lstm_units_2": 32,           # LSTM layer 2 units
    "dropout_rate": 0.20,         # Dropout for regularization
    "epochs":       50,           # Training epochs (EarlyStopping applies)
    "batch_size":   32,           # Mini-batch size
    "target_col":   "Close",      # Target column to predict
}

print(f"\n[CONFIG] Ticker: {CONFIG['ticker']} | Window: {CONFIG['window_size']} days")
print(f"[CONFIG] Train/Test Split: {int(CONFIG['train_ratio']*100)}% / {int((1-CONFIG['train_ratio'])*100)}%")


# ─────────────────────────────────────────────
# STEP 3 — DATA INGESTION (PUBLIC API)
# ─────────────────────────────────────────────
print("\n[STEP 3] Fetching OHLCV data from Yahoo Finance API...")

ticker_obj = yf.Ticker(CONFIG["ticker"])
df = ticker_obj.history(start=CONFIG["start_date"], end=CONFIG["end_date"])

# Select OHLCV columns
df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
df.dropna(inplace=True)
df.index = pd.to_datetime(df.index)

print(f"         Records fetched : {len(df)}")
print(f"         Date range      : {df.index[0].date()} → {df.index[-1].date()}")
print(f"         Columns         : {list(df.columns)}")
print(df.tail(3).to_string())


# ─────────────────────────────────────────────
# STEP 4 — FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[STEP 4] Engineering additional technical features...")

# Moving averages
df["MA_7"]  = df["Close"].rolling(7).mean()
df["MA_21"] = df["Close"].rolling(21).mean()

# Relative Strength Index (RSI)
delta = df["Close"].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
rs    = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

# Daily return
df["Return"] = df["Close"].pct_change()

# Drop NaN rows from rolling calculations
df.dropna(inplace=True)

print(f"         Features after engineering: {list(df.columns)}")
print(f"         Dataset shape : {df.shape}")


# ─────────────────────────────────────────────
# STEP 5 — PREPROCESSING & NORMALIZATION
# ─────────────────────────────────────────────
print("\n[STEP 5] Normalizing data with MinMaxScaler [0, 1]...")

feature_cols = ["Open", "High", "Low", "Close", "Volume",
                "MA_7", "MA_21", "RSI", "Return"]
target_col   = CONFIG["target_col"]
target_idx   = feature_cols.index(target_col)

data_arr = df[feature_cols].values

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data_arr)

# Separate scaler for inverse transforming predictions on Close only
close_scaler = MinMaxScaler()
close_scaler.fit_transform(df[[target_col]].values)

print(f"         Scaled shape : {data_scaled.shape}")
print(f"         Target index : [{target_idx}] → '{target_col}'")


# ─────────────────────────────────────────────
# STEP 6 — SLIDING WINDOW DATASET CREATION
# ─────────────────────────────────────────────
print(f"\n[STEP 6] Applying sliding window technique (window = {CONFIG['window_size']} days)...")

def create_sequences(data, window_size, target_idx):
    """
    Converts time-series data into (X, y) sequences.
    X shape: (samples, window_size, n_features)
    y shape: (samples,) — next-day Close price
    """
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, :])    # All features
        y.append(data[i, target_idx])            # Next-day Close
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, CONFIG["window_size"], target_idx)

# Train / test split (chronological — no shuffle)
split = int(len(X) * CONFIG["train_ratio"])
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"         Total sequences : {len(X)}")
print(f"         X_train shape  : {X_train.shape}  (samples, window, features)")
print(f"         X_test shape   : {X_test.shape}")


# ─────────────────────────────────────────────
# STEP 7 — BUILD LSTM MODEL
# ─────────────────────────────────────────────
print("\n[STEP 7] Building LSTM architecture...")

n_features = X_train.shape[2]

model = Sequential([
    LSTM(CONFIG["lstm_units_1"],
         return_sequences=True,
         input_shape=(CONFIG["window_size"], n_features),
         name="LSTM_Layer_1"),
    Dropout(CONFIG["dropout_rate"], name="Dropout_1"),

    LSTM(CONFIG["lstm_units_2"],
         return_sequences=False,
         name="LSTM_Layer_2"),
    Dropout(CONFIG["dropout_rate"], name="Dropout_2"),

    Dense(16, activation="relu", name="Dense_Hidden"),
    Dense(1,  activation="linear", name="Output")
], name="LSTM_StockForecaster")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mean_squared_error",
    metrics=["mae"]
)

model.summary()


# ─────────────────────────────────────────────
# STEP 8 — TRAIN MODEL
# ─────────────────────────────────────────────
print("\n[STEP 8] Training LSTM model...")

callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ModelCheckpoint("best_lstm_model.h5", monitor="val_loss",
                    save_best_only=True, verbose=0)
]

history = model.fit(
    X_train, y_train,
    epochs          = CONFIG["epochs"],
    batch_size      = CONFIG["batch_size"],
    validation_split= 0.10,
    callbacks       = callbacks,
    verbose         = 1
)

print(f"\n         Training complete. Epochs run: {len(history.history['loss'])}")


# ─────────────────────────────────────────────
# STEP 9 — EVALUATE MODEL
# ─────────────────────────────────────────────
print("\n[STEP 9] Evaluating model on test set...")

y_pred_scaled = model.predict(X_test, verbose=0).flatten()

# Inverse transform — reconstruct Close price scale
def inverse_close(scaled_vals, scaler, n_features, target_idx):
    """Inverse transform only the Close column."""
    dummy = np.zeros((len(scaled_vals), n_features))
    dummy[:, target_idx] = scaled_vals
    inversed = scaler.inverse_transform(dummy)
    return inversed[:, target_idx]

y_pred_actual = inverse_close(y_pred_scaled, scaler, n_features, target_idx)
y_test_actual = inverse_close(y_test,        scaler, n_features, target_idx)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae  = mean_absolute_error(y_test_actual, y_pred_actual)
mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100

print(f"\n  ┌─────────────────────────────────┐")
print(f"  │      MODEL PERFORMANCE METRICS  │")
print(f"  ├─────────────────────────────────┤")
print(f"  │  RMSE (%)  : {mape:.2f}%               │")
print(f"  │  RMSE ($)  : ${rmse:.4f}           │")
print(f"  │  MAE  ($)  : ${mae:.4f}           │")
print(f"  │  MAPE (%)  : {mape:.2f}%               │")
print(f"  └─────────────────────────────────┘")


# ─────────────────────────────────────────────
# STEP 10 — VISUALIZE RESULTS
# ─────────────────────────────────────────────
print("\n[STEP 10] Generating result plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(f"LSTM Stock Price Forecast — {CONFIG['ticker']}",
             fontsize=15, fontweight='bold', y=1.01)

# Plot 1: Actual vs Predicted
ax1 = axes[0, 0]
ax1.plot(y_test_actual, label="Actual Price",    color="#1f77b4", linewidth=1.5)
ax1.plot(y_pred_actual, label="Predicted Price", color="#ff7f0e", linewidth=1.5, linestyle="--")
ax1.set_title("Actual vs Predicted — Test Set")
ax1.set_xlabel("Trading Days")
ax1.set_ylabel("Stock Price (USD)")
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Training Loss
ax2 = axes[0, 1]
ax2.plot(history.history["loss"],     label="Train Loss", color="#2ca02c")
ax2.plot(history.history["val_loss"], label="Val Loss",   color="#d62728")
ax2.set_title("Training & Validation Loss (MSE)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: Residuals
ax3 = axes[1, 0]
residuals = y_test_actual - y_pred_actual
ax3.plot(residuals, color="#9467bd", linewidth=1)
ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax3.set_title("Residuals (Actual − Predicted)")
ax3.set_xlabel("Trading Days")
ax3.set_ylabel("Error (USD)")
ax3.grid(alpha=0.3)

# Plot 4: Metrics Bar
ax4 = axes[1, 1]
metrics = {"RMSE ($)": rmse, "MAE ($)": mae}
bars = ax4.bar(metrics.keys(), metrics.values(), color=["#1f77b4", "#ff7f0e"], width=0.4)
ax4.set_title(f"Error Metrics | MAPE: {mape:.2f}%")
ax4.set_ylabel("USD")
for bar in bars:
    ax4.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.1,
             f"${bar.get_height():.2f}", ha='center', fontsize=10)
ax4.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("stock_lstm_results.png", dpi=150, bbox_inches="tight")
print("         Plot saved: stock_lstm_results.png")
plt.show()


# ─────────────────────────────────────────────
# STEP 11 — SAVE MODEL & OUTPUTS
# ─────────────────────────────────────────────
print("\n[STEP 11] Saving model and predictions...")

model.save("stock_lstm_final.h5")

results_df = pd.DataFrame({
    "Actual_Close":    y_test_actual,
    "Predicted_Close": y_pred_actual,
    "Residual":        residuals
})
results_df.to_csv("predictions_output.csv", index=False)

print("         Model saved    : stock_lstm_final.h5")
print("         Predictions    : predictions_output.csv")
print("\n[DONE] Pipeline complete.")
print("=" * 55)
