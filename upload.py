import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, State, Output, no_update
import plotly.graph_objs as go
import io, base64
import os
from features import extract_features
from helper import build_table, build_layout

app = Dash(__name__)
app.title = "Sensor Dashboard"

df_global = pd.DataFrame()

# Layout (unchanged, just added export button)
app.layout = build_layout()

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
    print(df_global)
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
            
    else:
        return "No points selected."

    # Sort and arrays
    df_selected = df_selected.sort_values('time').reset_index(drop=True)
    y = df_selected['gas_resistance'].values.astype(float)
    t = df_selected['time'].values.astype(float)
    millis = df_selected['millis'].values.astype(int)

    if len(y) == 0:
        return "Selected points do not match any data."


    features_arr, feature_names = extract_features(
            sensor_id=selected_id,
            gas_arr=df_selected['gas_resistance'].values,
            temp_arr=df_selected['temperature'].values,
            pressure_arr=df_selected['pressure'].values,
            humidity_arr=df_selected['humidity'].values,
            millis_arr=df_selected['millis'].values
    )
        # Also show extracted features vector (pair names + values)
    feature_items = []
    for name, val in zip(feature_names, features_arr.flatten()):
        # some features may be NaN (should be turned into 0.0 by your extract_features), but format safely:
        try:
            feature_items.append(html.Li(f"{name}: {val:.2f}"))
        except Exception:
            feature_items.append(html.Li(f"{name}: {val}"))

    return html.Div([
        html.H4("Extracted Features:"),
        html.Ul(feature_items)
    ]), build_table(df_selected)    

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
        print(dropdown_options)
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

        features_arr, feature_names = extract_features(
            sensor_id=sensor_id,
            gas_arr=df_selected['gas_resistance'].values,
            temp_arr=df_selected['temperature'].values,
            pressure_arr=df_selected['pressure'].values,
            humidity_arr=df_selected['humidity'].values,
            millis_arr=df_selected['millis'].values,   # pass ms -> extract_features computes t from this
        )
        

        if not features_arr.any():
            return "No data to export."

        # Add label column
        feature_names_with_label = feature_names + ['label']
        
        features_arr_with_label = np.array(
            features_arr.tolist() + [curve_label],
            dtype=object
        )
        print(features_arr_with_label)
        
        df_export = pd.DataFrame([features_arr_with_label], columns=feature_names_with_label)

        file_exists = os.path.isfile("data.csv")
        if file_exists:
            df_export.to_csv("data.csv", mode='a', header=False, index=False)
        else:
            df_export.to_csv("data.csv", mode='w', header=True, index=False)

    return f"Exported {len(rows_to_export)} sensors to 'data.csv'."


if __name__ == '__main__':
    app.run(debug=True, port=8051)
