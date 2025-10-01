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
import numpy as np
import pandas as pd
from features import extract_features

# ---------------- CONFIG ----------------
SERIAL_PORT = "/dev/cu.usbserial-0289722F"
BAUDRATE = 115200
MAX_BUFFER_LINES = 5000
SEGMENT_SIZE = 20
THRESHOLD = 20
CONFIDENCE_THRESHOLD = 0.51   # how much confidence to show the current prediction
DROP_THRESHOLD_PERCENT = 0.2 # how much the baseline drops to start analyze
POST_BASELINE_POINTS = 100  # number of points to continue prediction after baseline
PREDICTION_START_SEGMENTS = 50 # How many points to wait before starting prediction

serial_lock = threading.Lock()

# Buffers per sensor ID
post_baseline_counter = defaultdict(int)
serial_buffer = defaultdict(lambda: deque(maxlen=MAX_BUFFER_LINES))
segment_buffer = defaultdict(lambda: deque(maxlen=SEGMENT_SIZE))
predictions_buffer = defaultdict(lambda: deque(maxlen=MAX_BUFFER_LINES))
current_prediction = defaultdict(lambda: None)
recent_predictions = defaultdict(list)
current_confidence = defaultdict(lambda: None)
collecting_segment = defaultdict(lambda: False)
segment_values = defaultdict(list)
segment_values_h = defaultdict(list)
segment_values_t = defaultdict(list)
segment_values_p = defaultdict(list)
segment_values_m = defaultdict(list)
baseline_value = defaultdict(lambda: None)
drop_start_value = defaultdict(lambda: None)
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

            # Initialize for this iteration
            pred_label = None
            raw_confidence = 0.0            

            parts = line.split(",")
            if len(parts) < 10:
                continue
            sensor_id = parts[0]
            gas_resistance = float(parts[8])
            temperature = float(parts[5])
            pressure = float(parts[6])
            humidity = float(parts[7])
            status = parts[9]
            gas_index = int(parts[3])
            mes_index = int(parts[4])
            index = int(parts[1])
            millis = int(parts[2])

            # Append to serial buffer for plotting
            with serial_lock:
                serial_buffer[sensor_id].append({"millis": millis, "gas_resistance": gas_resistance})
                predictions_buffer[sensor_id].append({
                    "millis": millis,
                    "prediction": current_prediction[sensor_id] if current_prediction[sensor_id] else "None"
                })

            # Initialize baseline
            if baseline_value[sensor_id] is None:
                baseline_value[sensor_id] = gas_resistance

            # --- DROP DETECTION ---
            if not collecting_segment[sensor_id]:
                drop_percent = (baseline_value[sensor_id] - gas_resistance) / baseline_value[sensor_id] * 100
                print(f"[{sensor_id}] Drop: {drop_percent:.2f}% (Threshold: {DROP_THRESHOLD_PERCENT*100:.0f}%)")

                if drop_percent >= DROP_THRESHOLD_PERCENT * 100:
                    collecting_segment[sensor_id] = True
                    # pegar resistência imediatamente antes do drop
                    drop_start_value[sensor_id] = serial_buffer[sensor_id][-2]["gas_resistance"] if len(serial_buffer[sensor_id]) >= 2 else gas_resistance
                    segment_values[sensor_id] = [drop_start_value[sensor_id], gas_resistance]
                    segment_values_t[sensor_id] = [drop_start_value[sensor_id], temperature]
                    segment_values_h[sensor_id] = [drop_start_value[sensor_id], humidity]
                    segment_values_p[sensor_id] = [drop_start_value[sensor_id], pressure]
                    segment_values_m[sensor_id] = [drop_start_value[sensor_id], millis]

            else:
                # Keep collecting the full drop/recovery segment
                segment_values[sensor_id].append(gas_resistance)
                segment_values_t[sensor_id].append(temperature)
                segment_values_h[sensor_id].append(humidity)
                segment_values_p[sensor_id].append(pressure)
                segment_values_m[sensor_id].append(millis)

                # Only start prediction after PREDICTION_START_SEGMENTS points
                if len(segment_values[sensor_id]) >= PREDICTION_START_SEGMENTS:
                    # features = extract_features_segment(segment_values[sensor_id], sensor_id,)
                    features, _ = extract_features(
                        sensor_id,
                        gas_arr=segment_values[sensor_id],
                        temp_arr=segment_values_t[sensor_id],
                        humidity_arr=segment_values_h[sensor_id],
                        pressure_arr=segment_values_p[sensor_id],
                        millis_arr=segment_values_m[sensor_id]
                    )
                    features = features.reshape(1, -1)
                    try:
                        pred_label = rf_model.predict(features)[0]
                        raw_confidence = float(np.max(rf_model.predict_proba(features)[0])) if hasattr(rf_model, "predict_proba") else 1.0
                    except Exception as e:
                        pred_label = None
                        raw_confidence = 0.0
                        print(e)

                    # current_prediction[sensor_id] = pred_label
                    # current_confidence[sensor_id] = raw_confidence

                # --- THRESHOLD & CONFIDENCE LOGIC ---
                    if raw_confidence >= CONFIDENCE_THRESHOLD and pred_label is not None:
                        recent_predictions[sensor_id].append(pred_label)
                        if len(recent_predictions[sensor_id]) > THRESHOLD:
                            recent_predictions[sensor_id].pop(0)

                        # Only update current_prediction if last THRESHOLD predictions are identical
                        if len(recent_predictions[sensor_id]) == THRESHOLD and len(set(recent_predictions[sensor_id])) == 1:
                            current_prediction[sensor_id] = pred_label
                            current_confidence[sensor_id] = raw_confidence
                    else:
                        pred_label = None  # Ignore low-confidence predictions   
                # Continue for POST_BASELINE_POINTS after reaching baseline
                if gas_resistance >= baseline_value[sensor_id]:
                    post_baseline_counter[sensor_id] += 1
                    if post_baseline_counter[sensor_id] >= POST_BASELINE_POINTS:
                        # Reset everything after threshold points
                        collecting_segment[sensor_id] = False
                        segment_values[sensor_id] = []
                        current_prediction[sensor_id] = None 
                        segment_values_t[sensor_id] = None
                        segment_values_h[sensor_id] = None
                        segment_values_p[sensor_id] = None
                        segment_values_m[sensor_id] = None
                        current_confidence[sensor_id] = None
                        post_baseline_counter[sensor_id] = 0
                else:
                    # Still below baseline → reset counter
                    post_baseline_counter[sensor_id] = 0

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
            csv_columns_full = list(row.keys())
            with open(csv_file, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_columns_full)
                writer.writerow(row)
            if collecting_segment[sensor_id]:
                display_label = "Analyzing..."
            else:
                display_label = "OK"

            # ---------------- PRINT OUTPUT ----------------
            print(f"[{sensor_id}] RAW={pred_label} | CURRENT={current_prediction[sensor_id]} "
                  f"(conf={current_confidence[sensor_id] if current_confidence[sensor_id] else 0:.2f})"
                  f"Threshold {len(recent_predictions[sensor_id])} - {THRESHOLD}")

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
        # preds = {sid: pd.DataFrame(list(predictions_buffer[sid])) for sid in sensor_ids}

    if not sensor_ids:
        return ["No sensors"], go.Figure()

    color_map = {
        "ar": "#10b981",
        "cigarro": "#ef4444",
        "alcool": "#3b82f6",
        "OK" : "#0c480c",
        None: "#9ca3af"
    }

    cards = []
    for sid in sensor_ids:
        # Use latest prediction if available
        pred_label = current_prediction[sid]
        pred_conf = current_confidence[sid] if current_confidence[sid] else 0.0

        if collecting_segment[sid]:
            if pred_label is not None:
                display_text = f"{pred_label} ({pred_conf:.2f})"
                bg_color = color_map.get(pred_label, "#fbbf24")
            else:
                display_text = "Analyzing..."
                bg_color = "#fbbf24"  # fallback
        else:
            if pred_label is None:
                display_text = "OK"
                bg_color = color_map.get("OK", "#0c480c")
            else:
                display_text = f"{pred_label} ({pred_conf:.2f})"
                bg_color = color_map.get(pred_label, "#9ca3af")

        cards.append(html.Div([
            html.H4(f"Sensor {sid}", style={"color":"white"}),
            html.Div(display_text,
                    style={"backgroundColor": bg_color,"color":"white","padding":"6px",
                            "borderRadius":"6px","textAlign":"center","fontSize":"16px"})
        ], style={"backgroundColor":"#222","padding":"10px","borderRadius":"8px"}))



    rows = int(np.ceil(len(sensor_ids)/2))
    cols = 2 if len(sensor_ids) > 1 else 1
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"Sensor {sid}" for sid in sensor_ids])
    for idx, sid in enumerate(sensor_ids):
        df = dfs[sid]
        if df.empty:
            continue

        # add timestamp for plotting
        df["timestamp"] = start_time + pd.to_timedelta(df["millis"], unit="ms")

        row = idx // cols + 1
        col = idx % cols + 1

        # --- main line trace ---
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["gas_resistance"],
            mode="lines", name=f"Gas {sid}",
            line=dict(color="#00FFFF")
        ), row=row, col=col)

        # --- highlight drop segment ---
        if segment_values.get(sid):
            seg_y = segment_values[sid]
            seg_x = df["timestamp"].iloc[-len(seg_y):]  # pegar os timestamps correspondentes
            seg_color = color_map.get(current_prediction[sid], "#fbbf24")
            fig.add_trace(go.Scatter(
                x=seg_x,
                y=seg_y,
                mode="markers+lines",
                name=f"Segment {sid}",
                line=dict(color=seg_color, dash="dot"),
                marker=dict(size=6, color=seg_color, symbol="circle"),
                showlegend=True
            ), row=row, col=col)
        # # --- highlight last prediction segment ---
        # if len(df) >= SEGMENT_SIZE:
        #     segment_df = df.iloc[-SEGMENT_SIZE:]
        #     seg_color = color_map.get(current_prediction[sid], "#fbbf24")  # orange fallback
        #     fig.add_trace(go.Scatter(
        #         x=segment_df["timestamp"], y=segment_df["gas_resistance"],
        #         mode="markers+lines",
        #         name=f"Segment {sid}",
        #         line=dict(color=seg_color, dash="dot"),
        #         marker=dict(size=6, color=seg_color, symbol="circle"),
        #         showlegend=True
        #     ), row=row, col=col)

    # for idx, sid in enumerate(sensor_ids):
    #     df = dfs[sid]
    #     if df.empty:
    #         continue
    #     df["timestamp"] = start_time + pd.to_timedelta(df["millis"], unit="ms")
    #     row = idx//cols + 1
    #     col = idx%cols + 1
    #     fig.add_trace(go.Scatter(
    #         x=df["timestamp"], y=df["gas_resistance"], mode="lines", name=f"Gas {sid}", line=dict(color="#00FFFF")
    #     ), row=row, col=col)

    fig.update_layout(template="plotly_dark", hovermode="x unified", height=800)
    fig.update_yaxes(title_text="Gas Resistance (Ω, log)", type="log")
    fig.update_xaxes(title_text="Time")

    return cards, fig

# ---------------- MAIN ----------------
if __name__ == "__main__":
    threading.Thread(target=serial_reader, daemon=True).start()
    app.run(debug=True, use_reloader=False)
