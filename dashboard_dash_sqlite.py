import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sqlite3
import datetime

# Database connection
DB_FILE = 'air_quality.sqlite'
TABLE = 'air_quality'

def get_connection():
    return sqlite3.connect(DB_FILE)

def get_sites():
    with get_connection() as conn:
        sites = pd.read_sql(f"SELECT DISTINCT site FROM {TABLE} ORDER BY site", conn)['site'].tolist()
    return sites

def get_date_range():
    with get_connection() as conn:
        min_date, max_date = pd.read_sql(f"SELECT MIN(datetime) as min_d, MAX(datetime) as max_d FROM {TABLE}", conn).iloc[0]
    return pd.to_datetime(min_date).date(), pd.to_datetime(max_date).date()

def get_pollutants():
    return ['pm25', 'pm10', 'no2']

# Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Urban Air Quality Dashboard (Dash/SQLite)'

sites = get_sites()
min_date, max_date = get_date_range()
pollutant_cols = get_pollutants()

app.layout = dbc.Container([
    dcc.Store(id='csv-data'),
    dcc.Download(id='download-data'),
    html.Button("📥 Download CSV", id="btn-download", n_clicks=0, style={'margin': '1rem 0'}),
    html.H1('🌫️ Urban Air Quality Dashboard (Dash/SQLite)', className='text-center my-4'),
    dbc.Row([
        dbc.Col([
            html.Label('Select Date Range'),
            dcc.DatePickerRange(
                id='date-range',
                min_date_allowed=min_date,
                max_date_allowed=max_date,
                start_date=min_date,
                end_date=max_date
            ),
            html.Br(), html.Br(),
            html.Label('Select Sites'),
            dcc.Dropdown(
                id='site-dropdown',
                options=[{'label': s, 'value': s} for s in sites],
                value=sites,
                multi=True
            ),
            html.Br(),
            html.Label('Select Pollutants'),
            dcc.Dropdown(
                id='pollutant-dropdown',
                options=[{'label': p, 'value': p} for p in pollutant_cols],
                value=pollutant_cols,
                multi=True
            ),
            html.Br(),
            html.Label('Time Aggregation'),
            dcc.Dropdown(
                id='agg-dropdown',
                options=[{'label': x, 'value': x} for x in ['Hourly', 'Daily', 'Weekly', 'Monthly']],
                value='Daily',
                clearable=False
            ),
        ], width=3),
        dbc.Col([
            dbc.Row([
                dbc.Col(html.Div(id='kpi-metrics'), width=12)
            ]),
            html.Hr(),
            dcc.Graph(id='time-series-chart'),
            dbc.Row([
                dbc.Col(dcc.Graph(id='site-comparison-chart'), width=6),
                dbc.Col([
                    html.Label('Select pollutant for heatmap:'),
                    dcc.Dropdown(
                        id='heatmap-pollutant-dropdown',
                        options=[{'label': p, 'value': p} for p in pollutant_cols],
                        value=pollutant_cols[0] if pollutant_cols else None,
                        clearable=False
                    ),
                    dcc.Graph(id='hourly-heatmap')
                ], width=6)
            ]),
            html.Hr(),
            dcc.Graph(id='correlation-chart'),
            html.Hr(),
            html.Div(id='data-export'),
            html.Hr(),
            dbc.Collapse([
                html.H5('💡 Air Quality Facts & Information'),
                html.Ul([
                    html.Li('🌍 WHO Air Quality Guidelines:'),
                    html.Ul([
                        html.Li('PM2.5: Annual mean ≤ 15 µg/m³, 24-hour mean ≤ 45 µg/m³'),
                        html.Li('PM10: Annual mean ≤ 45 µg/m³, 24-hour mean ≤ 50 µg/m³'),
                        html.Li('NO₂: Annual mean ≤ 10 µg/m³, 24-hour mean ≤ 25 µg/m³'),
                    ]),
                    html.Li('🔬 What are these pollutants?'),
                    html.Ul([
                        html.Li('PM2.5: Fine particulate matter ≤ 2.5 micrometers'),
                        html.Li('PM10: Coarse particulate matter ≤ 10 micrometers'),
                        html.Li('NO₂: Nitrogen dioxide, mainly from vehicles/industry'),
                    ]),
                    html.Li('⚕️ Health Impacts:'),
                    html.Ul([
                        html.Li('Poor air quality can cause respiratory/cardiovascular issues'),
                        html.Li('Children, elderly, and those with health conditions are most vulnerable'),
                        html.Li('Long-term exposure reduces life expectancy'),
                    ]),
                ])
            ], id='facts-collapse', is_open=True)
        ], width=9)
    ])
], fluid=True)

# Helper: Query filtered data
def query_data(start_date, end_date, sites, pollutants):
    site_list = ','.join(['?']*len(sites))
    poll_cols = ', '.join(pollutants)
    sql = f"SELECT site, datetime, {poll_cols} FROM {TABLE} WHERE date(datetime) >= ? AND date(datetime) <= ? AND site IN ({site_list})"
    params = [start_date, end_date] + sites
    print('SQL Query:', sql)
    print('Params:', params)
    with get_connection() as conn:
        df = pd.read_sql(sql, conn, params=params, parse_dates=['datetime'])
    print('Rows returned:', len(df))
    return df

# Callbacks
@app.callback(
    Output('kpi-metrics', 'children'),
    Output('time-series-chart', 'figure'),
    Output('site-comparison-chart', 'figure'),
    Output('hourly-heatmap', 'figure'),
    Output('correlation-chart', 'figure'),
    Output('data-export', 'children'),
    Output('csv-data', 'data'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
    Input('site-dropdown', 'value'),
    Input('pollutant-dropdown', 'value'),
    Input('agg-dropdown', 'value'),
    Input('heatmap-pollutant-dropdown', 'value')
)
def update_dashboard(start_date, end_date, selected_sites, selected_pollutants, aggregation, heatmap_pollutant):
    if not selected_sites or not selected_pollutants:
        return html.Div('No data for selected filters.'), go.Figure(), go.Figure(), go.Figure(), go.Figure(), '', ''
    df = query_data(start_date, end_date, selected_sites, selected_pollutants)
    if df.empty:
        return html.Div('No data for selected filters.'), go.Figure(), go.Figure(), go.Figure(), go.Figure(), '', ''
    # KPIs
    kpis = []
    avg_no2 = df['no2'].mean() if 'no2' in df.columns else np.nan
    avg_pm10 = df['pm10'].mean() if 'pm10' in df.columns else np.nan
    avg_pm25 = df['pm25'].mean() if 'pm25' in df.columns else np.nan
    poor_days = (df['pm25'] > 25).sum() if 'pm25' in df.columns else 0
    total_measurements = len(df[df['pm25'].notna()]) if 'pm25' in df.columns else 0
    percentage = (poor_days / total_measurements * 100) if total_measurements > 0 else 0
    kpis.append(html.Div([
        html.Div([
            html.H5('🔵 Avg NO₂ (µg/m³)'),
            html.H3(f"{avg_no2:.1f}" if not np.isnan(avg_no2) else "N/A")
        ], className='metric-container'),
        html.Div([
            html.H5('🟡 Avg PM10 (µg/m³)'),
            html.H3(f"{avg_pm10:.1f}" if not np.isnan(avg_pm10) else "N/A")
        ], className='metric-container'),
        html.Div([
            html.H5('🔴 Avg PM2.5 (µg/m³)'),
            html.H3(f"{avg_pm25:.1f}" if not np.isnan(avg_pm25) else "N/A")
        ], className='metric-container'),
        html.Div([
            html.H5('⚠️ Poor Air Quality'),
            html.H3(f"{poor_days} readings"),
            html.P(f"{percentage:.1f}% of total")
        ], className='metric-container'),
    ], style={'display': 'flex', 'gap': '2rem'}) )
    # Time Series
    df_agg = df.copy()
    if aggregation == "Daily":
        df_agg['date'] = df_agg['datetime'].dt.date
        df_agg = df_agg.groupby(['date', 'site'])[selected_pollutants].mean().reset_index()
        df_agg['datetime'] = pd.to_datetime(df_agg['date'])
    elif aggregation == "Weekly":
        df_agg['week'] = df_agg['datetime'].dt.to_period('W')
        df_agg = df_agg.groupby(['week', 'site'])[selected_pollutants].mean().reset_index()
        df_agg['datetime'] = df_agg['week'].dt.start_time
    elif aggregation == "Monthly":
        df_agg['month'] = df_agg['datetime'].dt.to_period('M')
        df_agg = df_agg.groupby(['month', 'site'])[selected_pollutants].mean().reset_index()
        df_agg['datetime'] = df_agg['month'].dt.start_time
    fig_ts = go.Figure()
    colors = {'no2': '#1f77b4', 'pm10': '#ff7f0e', 'pm25': '#d62728'}
    for pollutant in selected_pollutants:
        if pollutant in df_agg.columns:
            df_plot = df_agg.groupby('datetime')[pollutant].mean().reset_index()
            fig_ts.add_trace(go.Scatter(
                x=df_plot['datetime'],
                y=df_plot[pollutant],
                mode='lines+markers',
                name=pollutant.upper(),
                line=dict(color=colors.get(pollutant, '#000000'), width=3),
                marker=dict(size=6),
                hovertemplate=f'<b>{pollutant.upper()}</b><br>Date: %{{x}}<br>Value: %{{y:.1f}} µg/m³<extra></extra>'
            ))
    fig_ts.update_layout(
        title=f"📈 {aggregation} Air Quality Trends",
        xaxis_title="Date",
        yaxis_title="Concentration (µg/m³)",
        hovermode='x unified',
        height=500,
        showlegend=True,
        template="plotly_white"
    )
    # Site Comparison
    site_averages = df.groupby('site')[selected_pollutants].mean().reset_index()
    df_melted = site_averages.melt(
        id_vars='site', 
        value_vars=selected_pollutants,
        var_name='Pollutant', 
        value_name='Concentration'
    )
    fig_site = px.bar(
        df_melted,
        x='site',
        y='Concentration',
        color='Pollutant',
        title="📊 Average Pollutant Levels by Site",
        labels={'Concentration': 'Concentration (µg/m³)'},
        template="plotly_white",
        height=500
    )
    fig_site.update_layout(xaxis_tickangle=-45)
    # Hourly Heatmap
    df_heat = df.copy()
    df_heat['Hour'] = df_heat['datetime'].dt.hour
    df_heat['DayOfWeek'] = df_heat['datetime'].dt.day_name()
    heatmap_data = df_heat.groupby(['DayOfWeek', 'Hour'])[heatmap_pollutant].mean().reset_index()
    pivot_data = heatmap_data.pivot(index='DayOfWeek', columns='Hour', values=heatmap_pollutant)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_data = pivot_data.reindex(day_order)
    fig_heat = px.imshow(
        pivot_data,
        title=f"🕐 Hourly {heatmap_pollutant.upper()} Concentration Heatmap",
        labels=dict(x="Hour of Day", y="Day of Week", color=f"{heatmap_pollutant.upper()} (µg/m³)"),
        aspect="auto",
        color_continuous_scale="Reds",
        height=400
    )
    # Correlation
    from plotly.subplots import make_subplots
    fig_corr = make_subplots(
        rows=1, cols=3,
        subplot_titles=("PM2.5 vs PM10", "PM2.5 vs NO₂", "PM10 vs NO₂"),
        horizontal_spacing=0.1
    )
    if 'pm25' in df.columns and 'pm10' in df.columns:
        valid_data = df[['pm25', 'pm10']].dropna()
        if not valid_data.empty:
            fig_corr.add_trace(
                go.Scatter(x=valid_data['pm10'], y=valid_data['pm25'], mode='markers',
                          name='PM2.5 vs PM10', marker=dict(color='blue', opacity=0.6)),
                row=1, col=1
            )
    if 'pm25' in df.columns and 'no2' in df.columns:
        valid_data = df[['pm25', 'no2']].dropna()
        if not valid_data.empty:
            fig_corr.add_trace(
                go.Scatter(x=valid_data['no2'], y=valid_data['pm25'], mode='markers',
                          name='PM2.5 vs NO₂', marker=dict(color='red', opacity=0.6)),
                row=1, col=2
            )
    if 'pm10' in df.columns and 'no2' in df.columns:
        valid_data = df[['pm10', 'no2']].dropna()
        if not valid_data.empty:
            fig_corr.add_trace(
                go.Scatter(x=valid_data['no2'], y=valid_data['pm10'], mode='markers',
                          name='PM10 vs NO₂', marker=dict(color='green', opacity=0.6)),
                row=1, col=3
            )
    fig_corr.update_layout(
        title="🔗 Pollutant Correlation Analysis",
        height=500,
        showlegend=False,
        template="plotly_white"
    )
    fig_corr.update_xaxes(title_text="PM10 (µg/m³)", row=1, col=1)
    fig_corr.update_xaxes(title_text="NO₂ (µg/m³)", row=1, col=2)
    fig_corr.update_xaxes(title_text="NO₂ (µg/m³)", row=1, col=3)
    fig_corr.update_yaxes(title_text="PM2.5 (µg/m³)", row=1, col=1)
    fig_corr.update_yaxes(title_text="PM2.5 (µg/m³)", row=1, col=2)
    fig_corr.update_yaxes(title_text="PM10 (µg/m³)", row=1, col=3)
    # Data Export
    csv_buffer = df.to_csv(index=False)
    export_div = html.Div([
        html.H5('📤 Data Export'),
        html.P(f"Filtered Dataset: {len(df):,} records from {len(df['site'].unique())} sites")
    ])
    return kpis, fig_ts, fig_site, fig_heat, fig_corr, export_div, csv_buffer

@app.callback(
    Output("download-data", "data"),
    Input("btn-download", "n_clicks"),
    Input("csv-data", "data"),
    prevent_initial_call=True
)
def download_csv(n_clicks, csv_data):
    if n_clicks:
        return dict(content=csv_data, filename=f"air_quality_filtered_{datetime.date.today()}.csv")
    return dash.no_update

if __name__ == '__main__':
    app.run(debug=True) 