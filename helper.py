from dash import Dash, dcc, html, Input, State, Output, no_update
import plotly.graph_objs as go

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
            html.Td(f"{float(df_sel['gas_resistance'].iloc[i]):.3f}"),
            html.Td(f"{float(df_sel['temperature'].iloc[i]):.3f}")
        ]))
    if len(df_sel) > max_rows:
        rows.append(html.Tr([html.Td(f"... {len(df_sel)-max_rows} more rows", colSpan=4, style={'color':'white'})]))
    table_style = {'border': '1px solid #444', 'borderCollapse': 'collapse', 'color':'white'}
    return html.Table(header + rows, style=table_style)

def build_layout():
    return html.Div([
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