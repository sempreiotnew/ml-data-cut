import os
import serial
import threading
import sys
from collections import deque, defaultdict
import pandas as pd
import numpy as np
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from datetime import datetime
import joblib
from scipy.stats import skew, kurtosis
import csv

# ---------------- CONFIG ----------------
SERIAL_PORT = "/dev/cu.usbserial-0289714A"
BAUDRATE = 115200
MAX_BUFFER_LINES = 5000
SEGMENT_SIZE = 200
THRESHOLD = 50

serial_lock = threading.Lock()

# Buffers per sensor ID
serial_buffer = defaultdict(lambda: deque(maxlen=MAX_BUFFER_LINES))
segment_buffer = defaultdict(lambda: deque(maxlen=SEGMENT_SIZE))
predictions_buffer = defaultdict(lambda: deque(maxlen=MAX_BUFFER_LINES))
current_prediction = defaultdict(lambda: None)
current_confidence = defaultdict(lambda: None)

last_raw_prediction = defaultdict(lambda: None)
prediction_streak = defaultdict(int)

start_time = datetime.now()

# ---------------- LOAD MODEL ----------------
MODEL_FILE = "rf_sensor_model.pkl"
rf_model = joblib.load(MODEL_FILE)

# ---------------- CSV LOGGING ----------------
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file = f"predict_data_{timestamp_str}.csv"
csv_columns = ["id","index","millis","gas_index","mes_index","temperature","pressure","humidity","gas_resistance","status"]

# Initialize CSV
with open(csv_file, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_columns)
    writer.writeheader()

# ---------------- HELPER FUNCTIONS ----------------
def extract_features_segment(gas_arr):
    y = np.array(gas_arr, dtype=float)
    t = np.arange(len(y))

    baseline_vals = y[:min(5, len(y))]
    baseline_mean = float(np.mean(baseline_vals)) if len(baseline_vals) > 0 else float(y[0])

    min_idx = int(np.argmin(y))
    min_val = float(y[min_idx])
    drop_magnitude = baseline_mean - min_val
    drop_start_idx = 0
    drop_duration_s = float(max(0.0, t[min_idx] - t[drop_start_idx]))

    recovery_end_idx = len(y) - 1
    recovery_duration_s = float(max(0.0, t[recovery_end_idx] - t[min_idx]))
    total_response_time_s = float(max(0.0, t[recovery_end_idx] - t[drop_start_idx]))

    drop_rate = drop_magnitude / drop_duration_s if drop_duration_s > 0 else 0
    recovery_rate = (y[recovery_end_idx] - min_val) / recovery_duration_s if recovery_duration_s > 0 else 0

    drop_area = float(np.trapezoid(baseline_mean - y[drop_start_idx:min_idx + 1], t[drop_start_idx:min_idx + 1])) if min_idx > 0 else 0
    recovery_area = float(np.trapezoid(y[min_idx:recovery_end_idx + 1] - min_val, t[min_idx:recovery_end_idx + 1])) if recovery_end_idx > min_idx else 0

    skewness = float(pd.Series(y).skew())
    kurt = float(pd.Series(y).kurtosis())

    features = np.array([[baseline_mean, min_val, drop_magnitude, drop_duration_s,
                          recovery_duration_s, total_response_time_s, drop_rate,
                          recovery_rate, drop_area, recovery_area, skewness, kurt]])
    return features

# ---------------- SERIAL READER ----------------
def serial_reader():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.5)
    except Exception as e:
        print(f"[serial_reader] Error opening {SERIAL_PORT}: {e}")
        sys.exit(1)

    while True:
        try:
            line_bytes = ser.readline()
            if not line_bytes:
                continue
            line = line_bytes.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            try:
                parts = line.split(",")
                if len(parts) < 10:
                    continue
                sensor_id = parts[0]
                gas_resistance = float(parts[8])
                temperature = float(parts[5])
                pressure = float(parts[6])
                humidity = float(parts[7])
                status = parts[9]
                index = int(parts[1])
                gas_index = int(parts[3])
                mes_index = int(parts[4])
                millis = int(parts[2])
            except Exception as e:
                print("Parse error:", e)
                continue

            # Append to segment buffer
            segment_buffer[sensor_id].append(gas_resistance)
            if len(segment_buffer[sensor_id]) < SEGMENT_SIZE:
                continue

            # Extract features
            gas_arr = np.array(segment_buffer[sensor_id])
            features = extract_features_segment(gas_arr)

            # Predict
            pred_label = rf_model.predict(features)[0]
            raw_confidence = float(np.max(rf_model.predict_proba(features)[0])) if hasattr(rf_model, "predict_proba") else 1.0

            # Threshold logic
            if last_raw_prediction[sensor_id] == pred_label:
                prediction_streak[sensor_id] += 1
            else:
                prediction_streak[sensor_id] = 1
                last_raw_prediction[sensor_id] = pred_label

            if prediction_streak[sensor_id] >= THRESHOLD:
                current_prediction[sensor_id] = pred_label
                current_confidence[sensor_id] = raw_confidence
                prediction_streak[sensor_id] = 0

            # Store for plotting
            with serial_lock:
                serial_buffer[sensor_id].append({
                    "millis": millis,
                    "gas_resistance": gas_resistance
                })
                predictions_buffer[sensor_id].append({
                    "millis": millis,
                    "prediction": current_prediction[sensor_id] if current_prediction[sensor_id] else "None"
                })

            # ---------------- LOG TO CSV ----------------
            row = {
                "id": sensor_id,
                "index": index,
                "millis": millis,
                "gas_index": gas_index,
                "mes_index": mes_index,
                "temperature": temperature,
                "pressure": pressure,
                "humidity": humidity,
                "gas_resistance": gas_resistance,
                "status": status
            }
            with open(csv_file, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_columns)
                writer.writerow(row)

            print(f"[{sensor_id}] RAW={pred_label} | CURRENT={current_prediction[sensor_id]} (conf={current_confidence[sensor_id]:.2f})")

        except Exception as e:
            print("[serial_reader] Error:", e)

# ---------------- DASH APP ----------------
app = Dash(__name__)

app.layout = html.Div([
    html.H2("BME688 Live RF Prediction", style={"color":"white","textAlign":"center"}),
    html.Div(id="sensor-predictions", style={"display":"grid","gridTemplateColumns":"repeat(4, 1fr)","gap":"10px"}),
    dcc.Graph(id="live-graph", style={"height":"80vh"}),
    dcc.Interval(id="interval-refresh", interval=1000, n_intervals=0)
], style={"backgroundColor":"#111","padding":"20px"})

@app.callback(
    [Output("sensor-predictions","children"),
     Output("live-graph","figure")],
    Input("interval-refresh","n_intervals")
)
def update_dashboard(n):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    with serial_lock:
        sensor_ids = list(serial_buffer.keys())
        dfs = {sid: pd.DataFrame(list(serial_buffer[sid])) for sid in sensor_ids}
        preds = {sid: pd.DataFrame(list(predictions_buffer[sid])) for sid in sensor_ids}

    if not sensor_ids:
        return ["No sensors"], go.Figure()

    color_map = {
        "ar": "#10b981",
        "cigarro": "#ef4444",
        "alcool": "#3b82f6",
        None: "#9ca3af"
    }

    cards = []
    for sid in sensor_ids:
        pred_label = current_prediction[sid]
        pred_conf = current_confidence[sid]
        pred_text = f"{pred_label} ({pred_conf:.2f})" if pred_label else "None"
        bg_color = color_map.get(pred_label, "#9ca3af")

        cards.append(html.Div([
            html.H4(f"Sensor {sid}", style={"color":"white"}),
            html.Div(pred_text,
                    style={"backgroundColor": bg_color,"color":"white","padding":"6px",
                           "borderRadius":"6px","textAlign":"center","fontSize":"16px"})
        ], style={"backgroundColor":"#222","padding":"10px","borderRadius":"8px"}))

    rows = int(np.ceil(len(sensor_ids)/2))
    cols = 2 if len(sensor_ids) > 1 else 1
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"Sensor {sid}" for sid in sensor_ids])
    for idx, sid in enumerate(sensor_ids):
        df = dfs[sid]
        if df.empty: continue
        df["timestamp"] = start_time + pd.to_timedelta(df["millis"], unit="ms")
        row = idx//cols + 1
        col = idx%cols + 1
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["gas_resistance"], mode="lines", name=f"Gas {sid}", line=dict(color="#00FFFF")
        ), row=row, col=col)

    fig.update_layout(template="plotly_dark", hovermode="x unified", height=800)
    fig.update_yaxes(title_text="Gas Resistance (Ω, log)", type="log")
    fig.update_xaxes(title_text="Time")

    return cards, fig

# ---------------- MAIN ----------------
if __name__ == "__main__":
    threading.Thread(target=serial_reader, daemon=True).start()
    app.run(debug=True, use_reloader=False)
