import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, State, Output, no_update
import plotly.graph_objs as go
import io, base64
import os

app = Dash(__name__)
app.title = "Sensor Dashboard"

df_global = pd.DataFrame()

# Layout (unchanged, just added export button)
app.layout = html.Div([
    html.H1("Sensor Data Dashboard", style={'textAlign': 'center', 'color': 'white'}),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select a CSV File', style={'color': 'white', 'textDecoration': 'underline'})]),
        style={'width': '50%', 'height': '60px', 'lineHeight': '60px',
               'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
               'textAlign': 'center', 'margin': '10px auto', 'color': 'white',
               'backgroundColor': 'rgba(50,50,50,0.7)'},
        multiple=False
    ),
    html.Div(id='output-data-upload', style={'color': 'white', 'textAlign': 'center'}),
    html.Div([
        html.Label("Select Sensor ID:", style={'color': 'white'}),
        dcc.Dropdown(id='sensor-dropdown', placeholder="Select an ID", style={'color': 'black'})
    ], style={'width': '30%', 'margin': '20px auto'}),
    html.Div([
        dcc.Store(id='selected-points-store'),
        dcc.Graph(id='gas-graph', config={'displayModeBar': True, 'modeBarButtonsToAdd': ['select2d', 'lasso2d']}),
    ]),
    html.Div([
    html.Label("Label (class) for this curve:", style={'color': 'white'}),
        dcc.Input(id='curve-label-input', type='text', placeholder='ALCOHOL, CIGARRO, etc.', style={'color': 'black'})
    ], style={'width': '30%', 'margin': '10px auto'}),
    html.Div([
        html.Button("Export Selected Metrics to CSV", id="export-button", n_clicks=0, style={'margin': '10px'})
    ]),
    html.Div(id='selected-data-output', style={'color': 'white', 'margin': '20px'}),
    
], style={'backgroundColor': 'rgba(20,20,20,0.9)', 'minHeight': '100vh', 'padding': '20px'})

# ----------------------------
# CSV parsing and dropdown update (unchanged)
# ----------------------------
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return df
        else:
            return None
    except Exception as e:
        print(e)
        return None

@app.callback(
    Output('sensor-dropdown', 'options'),
    Output('sensor-dropdown', 'value'),
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_dropdown(contents, filename):
    global df_global
    if contents is None:
        return [], None, ""
    df_global = parse_contents(contents, filename)
    if df_global is None:
        return [], None, "Failed to load file."
    if 'id' not in df_global.columns:
        return [], None, "CSV missing 'id' column."
    ids = df_global['id'].unique()
    options = [{'label': str(i), 'value': i} for i in ids]
    first_id = ids[0] if len(ids) > 0 else None
    return options, first_id, f"File '{filename}' uploaded successfully. Found {len(ids)} unique sensors."

# ----------------------------
# Helper to extract valid selection info
# ----------------------------
def build_valid_selection_payload(selectedData):
    if not selectedData or not isinstance(selectedData, dict):
        return None

    pts = selectedData.get('points') or []
    indices = []
    for p in pts:
        try:
            if ('curveNumber' not in p) or (p.get('curveNumber') == 0):
                if 'pointIndex' in p:
                    indices.append(int(p['pointIndex']))
        except Exception:
            continue

    if indices:
        indices = sorted(list(dict.fromkeys(indices)))
        return {'indices': indices}

    rng = selectedData.get('range')
    if rng and isinstance(rng, dict) and rng.get('x'):
        try:
            x0 = float(rng['x'][0])
            x1 = float(rng['x'][1])
            return {'x_range': [min(x0, x1), max(x0, x1)]}
        except Exception:
            return None

    return None

# ----------------------------
# Store selected points
# ----------------------------
@app.callback(
    Output('selected-points-store', 'data'),
    Input('gas-graph', 'selectedData'),
    State('sensor-dropdown', 'value')
)
def store_selected_points(selectedData, current_sensor):
    payload = build_valid_selection_payload(selectedData)
    if payload is None:
        return no_update
    return {'sensor': current_sensor, **payload}

# ----------------------------
# Graph update callback (unchanged)
# ----------------------------
@app.callback(
    Output('gas-graph', 'figure'),
    Input('sensor-dropdown', 'value'),
    Input('selected-points-store', 'data')
)
def update_graph(selected_id, stored_selected):
    fig = go.Figure()
    fig.update_layout(title=f"Gas Resistance (ID {selected_id})" if selected_id is not None else "Gas Resistance",
                      plot_bgcolor='rgba(40,40,40,0.8)',
                      paper_bgcolor='rgba(50,50,50,0.8)',
                      font=dict(color='white'),
                      xaxis=dict(title='Time (s)', gridcolor='rgba(255,255,255,0.1)'),
                      yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                      hovermode="x unified",
                      height=750
                      )
    if selected_id is None or df_global.empty:
        return fig

    df_filtered = df_global[df_global['id'] == selected_id].copy()
    df_filtered['millis'] = pd.to_numeric(df_filtered.get('millis'), errors='coerce')
    df_filtered['gas_resistance'] = pd.to_numeric(df_filtered.get('gas_resistance'), errors='coerce')
    df_filtered = df_filtered.dropna(subset=['millis', 'gas_resistance'])
    if df_filtered.empty:
        return fig

    df_filtered = df_filtered.reset_index(drop=True)
    df_filtered['time'] = (df_filtered['millis'] - df_filtered['millis'].iloc[0]) / 1000.0

    fig.add_trace(go.Scatter(
        x=df_filtered['time'],
        y=df_filtered['gas_resistance'],
        mode='lines+markers',
        name='Gas Resistance',
        line=dict(color='red'),
        marker=dict(color='red')
    ))

    if stored_selected and stored_selected.get('sensor') == selected_id:
        df_selected = pd.DataFrame()
        if 'indices' in stored_selected:
            idxs = [i for i in stored_selected['indices'] if 0 <= i < len(df_filtered)]
            if idxs:
                df_selected = df_filtered.iloc[idxs].copy()
        elif 'x_range' in stored_selected:
            x_min, x_max = stored_selected['x_range']
            df_selected = df_filtered[(df_filtered['time'] >= x_min) & (df_filtered['time'] <= x_max)].copy()

        if not df_selected.empty:
            df_selected = df_selected.sort_values('time').reset_index(drop=True)
            y = df_selected['gas_resistance'].values
            t = df_selected['time'].values

            min_idx = int(np.argmin(y))
            if min_idx < len(y) - 1:
                max_idx = int(np.argmax(y[min_idx:])) + min_idx
            else:
                max_idx = min_idx

            fig.add_trace(go.Scatter(
                x=t,
                y=y,
                mode='lines',
                name='Selected Area',
                line=dict(color='yellow', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=[t[min_idx]], y=[y[min_idx]],
                mode='markers', name='Drop Min',
                marker=dict(color='blue', size=12, symbol='triangle-down')
            ))
            fig.add_trace(go.Scatter(
                x=[t[max_idx]], y=[y[max_idx]],
                mode='markers', name='Recovery Max',
                marker=dict(color='green', size=12, symbol='triangle-up')
            ))
    return fig

# Update metrics
@app.callback(
    Output('selected-data-output', 'children'),
    Input('sensor-dropdown', 'value'),
    Input('selected-points-store', 'data')
)
def update_metrics(selected_id, stored_selected):
    # If no data or no sensor selected -> show default
    if selected_id is None or df_global.empty:
        return "No points selected."

    # If there is no stored valid selection for current sensor, show message
    if not stored_selected or stored_selected.get('sensor') != selected_id:
        return "No points selected."

    # Build filtered dataframe
    df_filtered = df_global[df_global['id'] == selected_id].copy()
    df_filtered['millis'] = pd.to_numeric(df_filtered.get('millis'), errors='coerce')
    df_filtered['gas_resistance'] = pd.to_numeric(df_filtered.get('gas_resistance'), errors='coerce')
    df_filtered = df_filtered.dropna(subset=['millis', 'gas_resistance'])
    if df_filtered.empty:
        return "No numeric data for this sensor."

    df_filtered = df_filtered.reset_index(drop=True)
    df_filtered['time'] = (df_filtered['millis'] - df_filtered['millis'].iloc[0]) / 1000.0

    # Compose df_selected exactly using saved indices if available, otherwise using x-range
    if 'indices' in stored_selected:
        idxs = stored_selected['indices']
        idxs = [i for i in idxs if 0 <= i < len(df_filtered)]
        if not idxs:
            return "Selected points do not match any data."
        df_selected = df_filtered.iloc[idxs].copy().sort_values('time').reset_index(drop=True)
        # baseline candidates: up to 5 points before the first selected index, if any
        first_global_idx = idxs[0]
        baseline_candidates = df_filtered.iloc[max(0, first_global_idx - 5): first_global_idx]['gas_resistance'].values
    elif 'x_range' in stored_selected:
        x_min, x_max = stored_selected['x_range']
        df_selected = df_filtered[(df_filtered['time'] >= x_min) & (df_filtered['time'] <= x_max)].copy().sort_values('time').reset_index(drop=True)
        if df_selected.empty:
            return "Selected points do not match any data."
        # baseline candidates: up to 5 points BEFORE the selected region if available
        # find first index of selected region in df_filtered
        sel_idx_mask = (df_filtered['time'] >= x_min) & (df_filtered['time'] <= x_max)
        global_indices = np.where(sel_idx_mask)[0]
        if len(global_indices) > 0:
            first_global_idx = int(global_indices[0])
            baseline_candidates = df_filtered.iloc[max(0, first_global_idx - 5): first_global_idx]['gas_resistance'].values
        else:
            baseline_candidates = np.array([])
    else:
        return "No points selected."

    # Sort and arrays
    df_selected = df_selected.sort_values('time').reset_index(drop=True)
    y = df_selected['gas_resistance'].values.astype(float)
    t = df_selected['time'].values.astype(float)
    millis = df_selected['millis'].values.astype(int)

    if len(y) == 0:
        return "Selected points do not match any data."

    # --- Baseline Metrics ---
    # Prefer baseline candidates (up to 5 points before event). If not available, use first up to 5 points inside selection.
    if baseline_candidates is not None and len(baseline_candidates) >= 2:
        baseline_vals = np.array(baseline_candidates).astype(float)
    else:
        # fallback: first up to 5 points within selected window
        baseline_vals = y[:min(5, len(y))]

    baseline_mean = float(np.mean(baseline_vals)) if len(baseline_vals) > 0 else float(y[0])
    baseline_median = float(np.median(baseline_vals)) if len(baseline_vals) > 0 else float(y[0])
    baseline_std = float(np.std(baseline_vals)) if len(baseline_vals) > 0 else 0.0
    baseline_count = int(len(baseline_vals))

    # Minimum and drop magnitude
    min_idx = int(np.argmin(y))
    min_val = float(y[min_idx])
    drop_magnitude = float(baseline_mean - min_val)  # positive if drop occurred

    # --- Drop start detection (5% threshold) ---
    rel_threshold = 0.05
    drop_start_idx = None
    if abs(baseline_mean) > 1e-9:
        drop_mask = ((baseline_mean - y) / abs(baseline_mean)) >= rel_threshold
    else:
        # baseline too small, use absolute threshold relative to value range
        val_range = float(np.ptp(y)) if np.ptp(y) > 0 else 0.0
        drop_mask = (baseline_mean - y) >= (0.05 * val_range)

    # prefer the first index before or at min_idx that triggers mask
    candidates = np.where(drop_mask)[0]
    if candidates.size > 0:
        # pick the first candidate that occurs at or before min_idx if possible
        before_min = candidates[candidates <= min_idx]
        if before_min.size > 0:
            drop_start_idx = int(before_min[0])
        else:
            # otherwise take earliest candidate
            drop_start_idx = int(candidates[0])

    # If no threshold match found, attempt simple derivative-based detection:
    if drop_start_idx is None:
        dy = np.diff(y)
        # find first index where derivative is noticeably negative (e.g., less than -1% of baseline per point)
        thresh_deriv = -0.01 * (abs(baseline_mean) if abs(baseline_mean) > 0 else max(1.0, np.mean(np.abs(dy))))
        deriv_candidates = np.where(dy <= thresh_deriv)[0]
        if deriv_candidates.size > 0:
            drop_start_idx = int(deriv_candidates[0])
        else:
            # fallback to first selected point
            drop_start_idx = 0

    drop_start_time_s = float(t[drop_start_idx])
    drop_start_millis = int(millis[drop_start_idx])

    # --- Precise drop and recovery times ---
    t_min = float(t[min_idx])
    millis_min = int(millis[min_idx])

    # Recovery detection:
    recovery_threshold_value = baseline_mean - rel_threshold * abs(baseline_mean)  # within 5% of baseline
    # 90% recovery value (90% of drop recovered)
    recovery_90_value = min_val + 0.9 * drop_magnitude if drop_magnitude > 0 else baseline_mean

    # find first index after min_idx where y >= recovery_90_value
    idx_90 = None
    idx_recovery_end = None
    for j in range(min_idx, len(y)):
        if y[j] >= recovery_90_value:
            idx_90 = j
            break

    for j in range(min_idx, len(y)):
        if y[j] >= recovery_threshold_value:
            idx_recovery_end = j
            break

    # if no recovery end found within selection, use last point as partial recovery
    if idx_recovery_end is None:
        idx_recovery_end = len(y) - 1

    # if 90% not found, set as None (or last if partial)
    if idx_90 is None:
        idx_90_time = None
        idx_90_millis = None
    else:
        idx_90_time = float(t[idx_90])
        idx_90_millis = int(millis[idx_90])

    recovery_end_time_s = float(t[idx_recovery_end])
    recovery_end_millis = int(millis[idx_recovery_end])

    # durations
    drop_duration_s = float(max(0.0, t[min_idx] - t[drop_start_idx]))
    recovery_duration_s = float(max(0.0, t[idx_recovery_end] - t[min_idx]))
    total_response_time_s = float(max(0.0, t[idx_recovery_end] - t[drop_start_idx]))

    # --- Rate Metrics ---
    drop_rate = None
    if drop_duration_s > 0:
        drop_rate = float((baseline_mean - min_val) / drop_duration_s)  # units per second
    else:
        drop_rate = float(np.nan)

    recovery_rate = None
    if recovery_duration_s > 0:
        recovery_rate = float((y[idx_recovery_end] - min_val) / recovery_duration_s)
    else:
        recovery_rate = float(np.nan)

    curve_symmetry_time = None
    if recovery_duration_s > 0:
        curve_symmetry_time = float(drop_duration_s / recovery_duration_s) if recovery_duration_s != 0 else float('inf')

    # --- Area Metrics (AUC) ---
    # drop area: integrate baseline - y from drop_start_idx to min_idx
    drop_area = 0.0
    if min_idx >= drop_start_idx:
        drop_area = float(np.trapz(baseline_mean - y[drop_start_idx:min_idx + 1], t[drop_start_idx:min_idx + 1]))
    # recovery area: integrate y - min_val from min_idx to recovery_end_idx
    recovery_area = 0.0
    if idx_recovery_end >= min_idx:
        recovery_area = float(np.trapz(y[min_idx:idx_recovery_end + 1] - min_val, t[min_idx:idx_recovery_end + 1]))

    area_symmetry = None
    if recovery_area > 0:
        area_symmetry = float(drop_area / recovery_area)
    else:
        area_symmetry = float('inf') if drop_area > 0 else 0.0

    # --- Curve shape characteristics ---
    try:
        skewness = float(pd.Series(y).skew())
    except Exception:
        skewness = float('nan')
    try:
        kurtosis = float(pd.Series(y).kurtosis())
    except Exception:
        kurtosis = float('nan')

    # inflection count: count sign changes in derivative
    dy = np.diff(y)
    sign_changes = np.sum(np.abs(np.diff(np.sign(dy))) > 0)
    inflection_points = int(sign_changes)

    # --- Recovery analysis ---
    final_value = float(y[-1])
    recovery_percentage = 0.0
    if drop_magnitude > 0:
        recovery_percentage = float((final_value - min_val) / drop_magnitude * 100.0)
    else:
        recovery_percentage = 100.0 if final_value >= baseline_mean else 0.0

    recovery_magnitude = float(final_value - min_val)

    # timestamps (relative seconds and millis)
    timestamps_info = {
        "drop_start_time_s": drop_start_time_s,
        "drop_start_millis": drop_start_millis,
        "min_time_s": t_min,
        "min_millis": millis_min,
        "recovery_90_time_s": idx_90_time,
        "recovery_90_millis": idx_90_millis,
        "recovery_end_time_s": recovery_end_time_s,
        "recovery_end_millis": recovery_end_millis
    }

    # Build returned HTML (organized)
    return html.Div([
        html.H4("Baseline Metrics:"),
        html.Ul([
            html.Li(f"Baseline from {baseline_count} points — mean: {baseline_mean:.2f}, median: {baseline_median:.2f}, std: {baseline_std:.2f}"),
            html.Li(f"Minimum value in selected region: {min_val:.2f}"),
            html.Li(f"Drop magnitude (baseline - min): {drop_magnitude:.2f}")
        ]),
        html.H4("Timing Metrics:"),
        html.Ul([
            html.Li(f"Drop start (5% threshold): index {drop_start_idx}, time {drop_start_time_s:.2f} s ({drop_start_millis} ms)"),
            html.Li(f"Minimum point: index {min_idx}, time {t_min:.2f} s ({millis_min} ms)"),
            html.Li(f"90% recovery point: {('not reached' if idx_90 is None else f'index {idx_90}, time {idx_90_time:.2f} s ({idx_90_millis} ms)')}"),
            html.Li(f"Recovery end (within 5% of baseline or last point): index {idx_recovery_end}, time {recovery_end_time_s:.2f} s ({recovery_end_millis} ms)"),
            html.Li(f"Drop duration: {drop_duration_s:.2f} s"),
            html.Li(f"Recovery duration: {recovery_duration_s:.2f} s"),
            html.Li(f"Total response time: {total_response_time_s:.2f} s"),
        ]),
        html.H4("Rate Metrics:"),
        html.Ul([
            html.Li(f"Drop rate (units/s): {('N/A' if np.isnan(drop_rate) else f'{drop_rate:.6f}') }"),
            html.Li(f"Recovery rate (units/s): {('N/A' if np.isnan(recovery_rate) else f'{recovery_rate:.6f}') }"),
            html.Li(
                f"Curve symmetry (time ratio drop/recovery): "
                f"{'N/A' if curve_symmetry_time is None else ('inf' if curve_symmetry_time == float('inf') else f'{curve_symmetry_time:.2f}')}"
            )
            
        ]),
        html.H4("Area Metrics (AUC):"),
        html.Ul([
            html.Li(f"Drop AUC (baseline - curve): {drop_area:.6f} (units·s)"),
            html.Li(f"Recovery AUC (curve - min): {recovery_area:.6f} (units·s)"),
            html.Li(f"Area symmetry (drop/recovery): {('inf' if area_symmetry == float('inf') else f'{area_symmetry:.2f}')}")
        ]),
        html.H4("Curve Shape Characteristics:"),
        html.Ul([
            html.Li(f"Skewness: {('NaN' if np.isnan(skewness) else f'{skewness:.2f}') }"),
            html.Li(f"Kurtosis: {('NaN' if np.isnan(kurtosis) else f'{kurtosis:.2f}') }"),
            html.Li(f"Inflection points (approx): {inflection_points}")
        ]),
        html.H4("Recovery Analysis:"),
        html.Ul([
            html.Li(f"Recovery percentage (final point): {recovery_percentage:.2f}%"),
            html.Li(f"Recovery magnitude (final - min): {recovery_magnitude:.2f}")
        ]),
        html.H4("Key timestamps:"),
        html.Ul([
            html.Li(f"Drop start: {timestamps_info['drop_start_time_s']:.2f} s ({timestamps_info['drop_start_millis']} ms)"),
            html.Li(f"Minimum point: {timestamps_info['min_time_s']:.2f} s ({timestamps_info['min_millis']} ms)"),
            html.Li(f"Recovery 90%: {('not reached' if timestamps_info['recovery_90_time_s'] is None else f'{timestamps_info['recovery_90_time_s']:.2f} s ({timestamps_info['recovery_90_millis']} ms)')}"),
            html.Li(f"Recovery end: {timestamps_info['recovery_end_time_s']:.2f} s ({timestamps_info['recovery_end_millis']} ms)")
        ])
    ], style={'whiteSpace': 'pre-wrap', 'fontFamily': 'monospace'})

# ----------------------------
# EXPORT METRICS CALLBACK (modified)
# ----------------------------
@app.callback(
    Output('output-data-upload', 'children', allow_duplicate=True),
    Input('export-button', 'n_clicks'),
    State('selected-points-store', 'data'),
    State('sensor-dropdown', 'options'),
    State('curve-label-input', 'value'),  # <-- new label input
    prevent_initial_call=True
)
def export_metrics(n_clicks, stored_selected, dropdown_options, curve_label):
    if stored_selected is None or ('indices' not in stored_selected and 'x_range' not in stored_selected):
        return "No selection to export."

    if df_global.empty:
        return "No data loaded."

    if not curve_label:
        return "Please provide a label for this curve before exporting."

    rows_to_export = []

    for sensor_opt in dropdown_options:
        sensor_id = sensor_opt['value']
        df_filtered = df_global[df_global['id'] == sensor_id].copy()
        df_filtered['millis'] = pd.to_numeric(df_filtered.get('millis'), errors='coerce')
        df_filtered['gas_resistance'] = pd.to_numeric(df_filtered.get('gas_resistance'), errors='coerce')
        df_filtered = df_filtered.dropna(subset=['millis', 'gas_resistance'])
        if df_filtered.empty:
            continue
        df_filtered = df_filtered.reset_index(drop=True)
        df_filtered['time'] = (df_filtered['millis'] - df_filtered['millis'].iloc[0]) / 1000.0

        # determine selected points for this sensor
        if 'indices' in stored_selected:
            idxs = [i for i in stored_selected['indices'] if 0 <= i < len(df_filtered)]
            if not idxs:
                continue
            df_selected = df_filtered.iloc[idxs].copy()
        elif 'x_range' in stored_selected:
            x_min, x_max = stored_selected['x_range']
            df_selected = df_filtered[(df_filtered['time'] >= x_min) & (df_filtered['time'] <= x_max)].copy()
            if df_selected.empty:
                continue

        df_selected = df_selected.reset_index(drop=True)

        y = df_selected['gas_resistance'].values.astype(float)
        t = df_selected['time'].values.astype(float)
        baseline_vals = y[:min(5, len(y))]  # simple baseline for export

        baseline_mean = float(np.mean(baseline_vals)) if len(baseline_vals) > 0 else float(y[0])
        min_idx = int(np.argmin(y))
        min_val = float(y[min_idx])
        drop_magnitude = float(baseline_mean - min_val)
        drop_start_idx = 0
        drop_start_time_s = float(t[drop_start_idx])
        drop_duration_s = float(max(0.0, t[min_idx] - t[drop_start_idx]))
        recovery_end_idx = len(y) - 1
        recovery_duration_s = float(max(0.0, t[recovery_end_idx] - t[min_idx]))
        total_response_time_s = float(max(0.0, t[recovery_end_idx] - t[drop_start_idx]))

        drop_rate = (baseline_mean - min_val) / drop_duration_s if drop_duration_s > 0 else np.nan
        recovery_rate = (y[recovery_end_idx] - min_val) / recovery_duration_s if recovery_duration_s > 0 else np.nan
        drop_area = float(np.trapezoid(baseline_mean - y[drop_start_idx:min_idx + 1], t[drop_start_idx:min_idx + 1]))
        recovery_area = float(np.trapezoid(y[min_idx:recovery_end_idx + 1] - min_val, t[min_idx:recovery_end_idx + 1]))

        # --- new: curve shape metrics ---
        skewness = float(pd.Series(y).skew())
        kurtosis = float(pd.Series(y).kurtosis())

        row = {
            'sensor_id': sensor_id,
            'baseline_mean': baseline_mean,
            'min_val': min_val,
            'drop_magnitude': drop_magnitude,
            'drop_duration_s': drop_duration_s,
            'recovery_duration_s': recovery_duration_s,
            'total_response_time_s': total_response_time_s,
            'drop_rate': drop_rate,
            'recovery_rate': recovery_rate,
            'drop_area': drop_area,
            'recovery_area': recovery_area,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'label': curve_label  # <-- user-defined label for ML
        }
        rows_to_export.append(row)

    if not rows_to_export:
        return "No data to export."

    print(row)
    df_export = pd.DataFrame(rows_to_export)

    file_exists = os.path.isfile("data.csv")
    if file_exists:
        df_export.to_csv("data.csv", mode='a', header=False, index=False)
    else:
        df_export.to_csv("data.csv", mode='w', header=True, index=False)

    return f"Exported {len(rows_to_export)} sensors to 'data.csv'."

if __name__ == '__main__':
    app.run(debug=True, port=8051)
