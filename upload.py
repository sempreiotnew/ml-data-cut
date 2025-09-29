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

# ----------------------------
# Fingerprint computation helper (used for both UI and export)
# ----------------------------
def compute_fingerprint(t, y, millis=None, rel_threshold=0.05):
    """
    t: seconds array
    y: signal array (gas_resistance)
    millis: optional original millis
    returns: dict with fingerprint features
    """
    out = {}
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if millis is None:
        millis = (t * 1000).astype(int)
    else:
        millis = np.asarray(millis, dtype=int)

    n = len(y)
    out['n_points'] = int(n)
    if n == 0:
        return out

    # basic smoothing for detection only (do not change raw arrays)
    window = min(11, max(3, int(max(1, n * 0.02))))
    y_smooth = pd.Series(y).rolling(window=window, center=True, min_periods=1).median().values

    # baseline region: try up to 5 points before event, else first 5%/25% fallback
    baseline_n = min(max(5, int(0.05 * n)), max(5, int(0.25 * n)))
    baseline_vals = y[:baseline_n] if len(y) >= baseline_n else y[:max(1, len(y)//2)]
    baseline_mean = float(np.nanmean(baseline_vals)) if baseline_vals.size > 0 else float(y[0])
    baseline_median = float(np.nanmedian(baseline_vals)) if baseline_vals.size > 0 else float(y[0])
    baseline_std = float(np.nanstd(baseline_vals)) if baseline_vals.size > 0 else 0.0
    out.update({'baseline_mean': baseline_mean, 'baseline_median': baseline_median, 'baseline_std': baseline_std, 'baseline_count': int(len(baseline_vals))})

    # min using smooth for index robustness
    min_idx = int(np.argmin(y_smooth))
    min_val = float(y[min_idx])
    out.update({'min_idx': min_idx, 'min_val': min_val})

    drop_magnitude = float(baseline_mean - min_val)
    out['drop_magnitude'] = drop_magnitude

    # drop start detection: relative threshold on smoothed signal
    drop_start_idx = None
    if abs(baseline_mean) > 1e-9:
        drop_mask = ((baseline_mean - y_smooth) / abs(baseline_mean)) >= rel_threshold
    else:
        val_range = float(np.ptp(y_smooth)) if np.ptp(y_smooth) > 0 else 0.0
        drop_mask = (baseline_mean - y_smooth) >= (rel_threshold * max(1.0, val_range))

    candidates = np.where(drop_mask)[0]
    if candidates.size > 0:
        # choose earliest run that leads to min if possible
        before_min = candidates[candidates <= min_idx]
        if before_min.size > 0:
            # find contiguous runs and pick the run that begins earliest
            runs = np.split(before_min, np.where(np.diff(before_min) != 1)[0] + 1)
            drop_start_idx = int(runs[0][0])
        else:
            runs = np.split(candidates, np.where(np.diff(candidates) != 1)[0] + 1)
            drop_start_idx = int(runs[0][0])

    # derivative fallback
    if drop_start_idx is None:
        dy = np.diff(y_smooth)
        mean_abs_dy = float(np.mean(np.abs(dy))) if dy.size > 0 else 0.0
        thresh_deriv = -0.01 * max(abs(baseline_mean), mean_abs_dy, 1.0)
        deriv_candidates = np.where(dy <= thresh_deriv)[0]
        if deriv_candidates.size > 0:
            before_min = deriv_candidates[deriv_candidates <= max(0, min_idx - 1)]
            drop_start_idx = int(before_min[0]) if before_min.size > 0 else int(deriv_candidates[0])
        else:
            drop_start_idx = 0

    drop_start_idx = max(0, min(drop_start_idx, n-1))
    drop_start_time_s = float(t[drop_start_idx])
    drop_start_millis = int(millis[drop_start_idx])
    out.update({'drop_start_idx': drop_start_idx, 'drop_start_time_s': drop_start_time_s, 'drop_start_millis': drop_start_millis})

    # recovery detection (90% and full within rel_threshold)
    if abs(baseline_mean) > 1e-9:
        recovery_threshold_value = baseline_mean - rel_threshold * abs(baseline_mean)
    else:
        val_range = float(np.ptp(y_smooth)) if np.ptp(y_smooth) > 0 else 0.0
        recovery_threshold_value = baseline_mean - rel_threshold * max(1.0, val_range)

    idx_90 = None
    idx_recovery_end = None
    recovery_90_value = min_val + 0.9 * drop_magnitude if drop_magnitude > 0 else baseline_mean

    for j in range(min_idx, n):
        if y_smooth[j] >= recovery_90_value:
            idx_90 = int(j)
            break

    for j in range(min_idx, n):
        if y_smooth[j] >= recovery_threshold_value:
            idx_recovery_end = int(j)
            break

    if idx_recovery_end is None:
        idx_recovery_end = n - 1

    if idx_90 is None:
        idx_90_time = None
        idx_90_millis = None
    else:
        idx_90_time = float(t[idx_90])
        idx_90_millis = int(millis[idx_90])

    out.update({
        'min_time_s': float(t[min_idx]), 'min_millis': int(millis[min_idx]),
        'recovery_90_idx': idx_90, 'recovery_90_time_s': idx_90_time, 'recovery_90_millis': idx_90_millis,
        'recovery_end_idx': idx_recovery_end, 'recovery_end_time_s': float(t[idx_recovery_end]), 'recovery_end_millis': int(millis[idx_recovery_end])
    })

    # durations (in seconds because t is seconds)
    drop_duration_s = float(max(0.0, t[min_idx] - t[drop_start_idx]))
    recovery_duration_s = float(max(0.0, t[idx_recovery_end] - t[min_idx]))
    total_response_time_s = float(max(0.0, t[idx_recovery_end] - t[drop_start_idx]))
    out.update({'drop_duration_s': drop_duration_s, 'recovery_duration_s': recovery_duration_s, 'total_response_time_s': total_response_time_s})

    # rates
    drop_rate = (baseline_mean - min_val) / drop_duration_s if drop_duration_s > 0 else float('nan')
    recovery_rate = (y[idx_recovery_end] - min_val) / recovery_duration_s if recovery_duration_s > 0 else float('nan')
    out.update({'drop_rate': float(drop_rate), 'recovery_rate': float(recovery_rate)})

    # slope metrics (raw)
    dy_raw = np.diff(y)
    dt_raw = np.diff(t)
    slopes = dy_raw / np.where(dt_raw == 0, np.nan, dt_raw)
    # negative slopes (drop) and positive slopes (recovery)
    max_negative_slope = float(np.nanmin(slopes)) if slopes.size > 0 else float('nan')
    max_positive_slope = float(np.nanmax(slopes)) if slopes.size > 0 else float('nan')
    mean_slope = float(np.nanmean(slopes)) if slopes.size > 0 else float('nan')
    out.update({'max_negative_slope': max_negative_slope, 'max_positive_slope': max_positive_slope, 'mean_slope': mean_slope})

    # areas using new numpy trapezoid with explicit x= (seconds)
    try:
        if min_idx >= drop_start_idx:
            drop_area = float(np.trapezoid(baseline_mean - y[drop_start_idx:min_idx + 1], x=t[drop_start_idx:min_idx + 1]))
        else:
            drop_area = 0.0
    except Exception:
        drop_area = 0.0
    try:
        if idx_recovery_end >= min_idx:
            recovery_area = float(np.trapezoid(y[min_idx:idx_recovery_end + 1] - min_val, x=t[min_idx:idx_recovery_end + 1]))
        else:
            recovery_area = 0.0
    except Exception:
        recovery_area = 0.0
    out.update({'drop_area': drop_area, 'recovery_area': recovery_area})
    out['area_symmetry'] = float(drop_area / recovery_area) if recovery_area > 0 else (float('inf') if drop_area > 0 else 0.0)

    # shape
    try:
        skewness = float(pd.Series(y).skew())
    except Exception:
        skewness = float('nan')
    try:
        kurtosis = float(pd.Series(y).kurtosis())
    except Exception:
        kurtosis = float('nan')
    out.update({'skewness': skewness, 'kurtosis': kurtosis})

    # inflection count (smoothed derivative)
    dy_s = np.diff(y_smooth)
    sign_changes = np.sum(np.abs(np.diff(np.sign(dy_s))) > 0) if dy_s.size > 1 else 0
    out['inflection_points'] = int(sign_changes)

    # final value & recovery percent
    final_value = float(y[-1])
    recovery_percentage = float((final_value - min_val) / drop_magnitude * 100.0) if drop_magnitude > 0 else (100.0 if final_value >= baseline_mean else 0.0)
    out.update({'final_value': final_value, 'recovery_percentage': recovery_percentage, 'recovery_magnitude': float(final_value - min_val)})

    # peak counts (local maxima) after min (recovery peaks) and before min (spikes)
    from numpy import sign
    peaks_after_min = 0
    peaks_before_min = 0
    if n >= 3:
        # compute simple local maxima on raw y
        for i in range(1, n-1):
            if y[i] > y[i-1] and y[i] > y[i+1]:
                if i > min_idx:
                    peaks_after_min += 1
                else:
                    peaks_before_min += 1
    out.update({'peaks_before_min': int(peaks_before_min), 'peaks_after_min': int(peaks_after_min)})

    # time to min fraction (normalized)
    total_dur = float(t[-1] - t[0]) if t[-1] > t[0] else float('nan')
    out['time_to_min_fraction'] = float((t[min_idx] - t[0]) / total_dur) if total_dur and not np.isnan(total_dur) else float('nan')

    return out

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

    # df_filtered = df_filtered.reset_index(drop=True)
    # # use seconds for all time metrics
    # df_filtered['time'] = (df_filtered['millis'] - df_filtered['millis'].iloc[0]) / 1000.0

    df_filtered = df_filtered.reset_index(drop=True)
    # KEEP absolute time in seconds
    df_filtered['time'] = df_filtered['millis'] / 1000.0
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

    # Baseline choices consistent with compute_fingerprint (UI shows same baseline used)
    if baseline_candidates is not None and len(baseline_candidates) >= 2:
        baseline_vals = np.array(baseline_candidates).astype(float)
    else:
        baseline_vals = y[:min(5, len(y))]

    baseline_mean = float(np.mean(baseline_vals)) if len(baseline_vals) > 0 else float(y[0])
    baseline_median = float(np.median(baseline_vals)) if len(baseline_vals) > 0 else float(y[0])
    baseline_std = float(np.std(baseline_vals)) if len(baseline_vals) > 0 else 0.0
    baseline_count = int(len(baseline_vals))

    # Compute fingerprint (this returns the canonical same values we will export)
    fingerprint = compute_fingerprint(t, y, millis=millis, rel_threshold=0.05)

    # Build table of selected points (cap rows to 200)
    def build_table(df_sel, max_rows=200):
        header = [html.Tr([html.Th("idx"), html.Th("millis"), html.Th("time_s"), html.Th("gas_resistance")], style={'color':'white'})]
        rows = []
        nrows = min(len(df_sel), max_rows)
        for i in range(nrows):
            rows.append(html.Tr([
                html.Td(str(i)),
                html.Td(str(int(df_sel['millis'].iloc[i]))),
                html.Td(f"{float(df_sel['time'].iloc[i]):.3f}"),
                html.Td(f"{float(df_sel['gas_resistance'].iloc[i]):.3f}")
            ]))
        if len(df_sel) > max_rows:
            rows.append(html.Tr([html.Td(f"... {len(df_sel)-max_rows} more rows", colSpan=4, style={'color':'white'})]))
        table_style = {'border': '1px solid #444', 'borderCollapse': 'collapse', 'color':'white'}
        return html.Table(header + rows, style=table_style)

    # Build returned HTML (organized) and ensure numeric formatting matches export
    return html.Div([
        html.H4("Baseline Metrics:"),
        html.Ul([
            html.Li(f"Baseline from {baseline_count} points — mean: {baseline_mean:.2f}, median: {baseline_median:.2f}, std: {baseline_std:.2f}"),
            html.Li(f"Minimum value in selected region: {fingerprint.get('min_val', float('nan')):.2f}"),
            html.Li(f"Drop magnitude (baseline - min): {fingerprint.get('drop_magnitude', float('nan')):.2f}")
        ]),
        html.H4("Timing Metrics:"),
        html.Ul([
            html.Li(f"Drop start: index {fingerprint.get('drop_start_idx')} , time {fingerprint.get('drop_start_time_s', 0.0):.3f} s ({fingerprint.get('drop_start_millis')} ms)"),
            html.Li(f"Minimum point: index {fingerprint.get('min_idx')}, time {fingerprint.get('min_time_s', 0.0):.3f} s ({fingerprint.get('min_millis')} ms)"),
            # html.Li(f"90% recovery point: {('not reached' if fingerprint.get('recovery_90_idx') is None else f'index {fingerprint.get(\"recovery_90_idx\")}, time {fingerprint.get(\"recovery_90_time_s\"):.3f} s ({fingerprint.get(\"recovery_90_millis\")} ms)')}"),
            html.Li(f"Recovery end (within 5% of baseline or last point): index {fingerprint.get('recovery_end_idx')}, time {fingerprint.get('recovery_end_time_s', 0.0):.3f} s ({fingerprint.get('recovery_end_millis')} ms)"),
            html.Li(f"Drop duration: {fingerprint.get('drop_duration_s', 0.0):.3f} s"),
            html.Li(f"Recovery duration: {fingerprint.get('recovery_duration_s', 0.0):.3f} s"),
            html.Li(f"Total response time: {fingerprint.get('total_response_time_s', 0.0):.3f} s"),
        ]),
        html.H4("Rate & Slope Metrics:"),
        html.Ul([
            html.Li(f"Drop rate (units/s): {('N/A' if np.isnan(fingerprint.get('drop_rate', np.nan)) else f'{fingerprint.get('drop_rate'):.6f}') }"),
            html.Li(f"Recovery rate (units/s): {('N/A' if np.isnan(fingerprint.get('recovery_rate', np.nan)) else f'{fingerprint.get('recovery_rate'):.6f}') }"),
            html.Li(f"Max negative slope (raw): {fingerprint.get('max_negative_slope'):.6f}"),
            html.Li(f"Max positive slope (raw): {fingerprint.get('max_positive_slope'):.6f}"),
            html.Li(f"Mean slope: {('NaN' if np.isnan(fingerprint.get('mean_slope', np.nan)) else f'{fingerprint.get('mean_slope'):.6f}')}")
        ]),
        html.H4("Area Metrics (AUC):"),
        html.Ul([
            html.Li(f"Drop AUC (baseline - curve): {fingerprint.get('drop_area', 0.0):.6f} (units·s)"),
            html.Li(f"Recovery AUC (curve - min): {fingerprint.get('recovery_area', 0.0):.6f} (units·s)"),
            html.Li(f"Area symmetry (drop/recovery): {('inf' if fingerprint.get('area_symmetry') == float('inf') else f'{fingerprint.get('area_symmetry'):.2f}')}")
        ]),
        html.H4("Shape & Peaks:"),
        html.Ul([
            html.Li(f"Skewness: {('NaN' if np.isnan(fingerprint.get('skewness', np.nan)) else f'{fingerprint.get('skewness'):.2f}') }"),
            html.Li(f"Kurtosis: {('NaN' if np.isnan(fingerprint.get('kurtosis', np.nan)) else f'{fingerprint.get('kurtosis'):.2f}') }"),
            html.Li(f"Inflection points (approx): {fingerprint.get('inflection_points')}"),
            html.Li(f"Peaks before min: {fingerprint.get('peaks_before_min')}, peaks after min: {fingerprint.get('peaks_after_min')}")
        ]),
        html.H4("Recovery Analysis:"),
        html.Ul([
            html.Li(f"Recovery percentage (final point): {fingerprint.get('recovery_percentage', 0.0):.2f}%"),
            html.Li(f"Recovery magnitude (final - min): {fingerprint.get('recovery_magnitude', 0.0):.6f}")
        ]),
        html.H4("Key timestamps:"),
        html.Ul([
            html.Li(f"Drop start: {fingerprint.get('drop_start_time_s', 0.0):.3f} s ({fingerprint.get('drop_start_millis')} ms)"),
            html.Li(f"Minimum point: {fingerprint.get('min_time_s', 0.0):.3f} s ({fingerprint.get('min_millis')} ms)"),
            html.Li(f"Recovery 90%: {('not reached' if fingerprint.get('recovery_90_time_s') is None else f'{fingerprint.get('recovery_90_time_s'):.3f} s ({fingerprint.get('recovery_90_millis')} ms)')}"),
            html.Li(f"Recovery end: {fingerprint.get('recovery_end_time_s', 0.0):.3f} s ({fingerprint.get('recovery_end_millis')} ms)")
        ]),
        html.H4("Selected points (table, first rows):"),
        build_table(df_selected)
    ], style={'whiteSpace': 'pre-wrap', 'fontFamily': 'monospace'})

# ----------------------------
# EXPORT METRICS CALLBACK (modified to export the exact same fingerprint fields)
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

    # For each sensor, build df_selected using the stored selection (same behaviour as UI)
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
            df_selected = df_filtered.iloc[idxs].copy().sort_values('time').reset_index(drop=True)
        elif 'x_range' in stored_selected:
            x_min, x_max = stored_selected['x_range']
            df_selected = df_filtered[(df_filtered['time'] >= x_min) & (df_filtered['time'] <= x_max)].copy().sort_values('time').reset_index(drop=True)
            if df_selected.empty:
                continue
        else:
            continue

        y = df_selected['gas_resistance'].values.astype(float)
        t = df_selected['time'].values.astype(float)
        millis = df_selected['millis'].values.astype(int)

        # baseline for export same as UI (first up to 5 points)
        baseline_vals = y[:min(5, len(y))] if len(y) > 0 else np.array([0.0])
        baseline_mean = float(np.mean(baseline_vals)) if len(baseline_vals) > 0 else float(y[0] if len(y)>0 else 0.0)

        fingerprint = compute_fingerprint(t, y, millis=millis, rel_threshold=0.05)

        row = {
            'sensor_id': sensor_id,
            'n_points': fingerprint.get('n_points'),
            'baseline_mean': fingerprint.get('baseline_mean'),
            'baseline_median': fingerprint.get('baseline_median'),
            'baseline_std': fingerprint.get('baseline_std'),
            'min_val': fingerprint.get('min_val'),
            'min_idx': fingerprint.get('min_idx'),
            'drop_magnitude': fingerprint.get('drop_magnitude'),
            'drop_start_idx': fingerprint.get('drop_start_idx'),
            'drop_start_time_s': fingerprint.get('drop_start_time_s'),
            'drop_duration_s': fingerprint.get('drop_duration_s'),
            'recovery_end_idx': fingerprint.get('recovery_end_idx'),
            'recovery_duration_s': fingerprint.get('recovery_duration_s'),
            'total_response_time_s': fingerprint.get('total_response_time_s'),
            'drop_rate': fingerprint.get('drop_rate'),
            'recovery_rate': fingerprint.get('recovery_rate'),
            'max_negative_slope': fingerprint.get('max_negative_slope'),
            'max_positive_slope': fingerprint.get('max_positive_slope'),
            'drop_area': fingerprint.get('drop_area'),
            'recovery_area': fingerprint.get('recovery_area'),
            'area_symmetry': fingerprint.get('area_symmetry'),
            'skewness': fingerprint.get('skewness'),
            'kurtosis': fingerprint.get('kurtosis'),
            'inflection_points': fingerprint.get('inflection_points'),
            'peaks_before_min': fingerprint.get('peaks_before_min'),
            'peaks_after_min': fingerprint.get('peaks_after_min'),
            'recovery_percentage': fingerprint.get('recovery_percentage'),
            'final_value': fingerprint.get('final_value'),
            'time_to_min_fraction': fingerprint.get('time_to_min_fraction'),
            'label': curve_label
        }
        rows_to_export.append(row)

    if not rows_to_export:
        return "No data to export."

    df_export = pd.DataFrame(rows_to_export)

    file_exists = os.path.isfile("data.csv")
    if file_exists:
        df_export.to_csv("data.csv", mode='a', header=False, index=False)
    else:
        df_export.to_csv("data.csv", mode='w', header=True, index=False)

    return f"Exported {len(rows_to_export)} sensors to 'data.csv'."

if __name__ == '__main__':
    app.run(debug=True, port=8051)
