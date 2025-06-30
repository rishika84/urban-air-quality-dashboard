import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import datetime

DB_FILE = 'air_quality.sqlite'
TABLE = 'air_quality'
MAX_POINTS = 1000  # Downsample if more than this many points
MAX_SITES = 3
MAX_DAYS = 30
WHO_THRESHOLDS = {'pm25': 25, 'pm10': 50, 'no2': 25}

@st.cache_data(show_spinner=False)
def get_sites():
    with sqlite3.connect(DB_FILE) as conn:
        sites = pd.read_sql(f"SELECT DISTINCT site FROM {TABLE} ORDER BY site", conn)['site'].tolist()
    return sites

@st.cache_data(show_spinner=False)
def get_date_range():
    with sqlite3.connect(DB_FILE) as conn:
        min_date, max_date = pd.read_sql(f"SELECT MIN(datetime) as min_d, MAX(datetime) as max_d FROM {TABLE}", conn).iloc[0]
    return pd.to_datetime(min_date).date(), pd.to_datetime(max_date).date()

@st.cache_data(show_spinner=False)
def get_pollutants():
    return ['pm25', 'pm10', 'no2']

@st.cache_data(show_spinner=False)
def query_data(start_date, end_date, sites, pollutants):
    site_list = ','.join(['?']*len(sites))
    poll_cols = ', '.join(pollutants)
    sql = f"SELECT site, datetime, {poll_cols} FROM {TABLE} WHERE date(datetime) >= ? AND date(datetime) <= ? AND site IN ({site_list})"
    params = [start_date, end_date] + sites
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql(sql, conn, params=params, parse_dates=['datetime'])
    return df

@st.cache_data(show_spinner=False)
def query_aggregated(start_date, end_date, sites, pollutants, aggregation):
    site_list = ','.join(['?']*len(sites))
    poll_exprs = ', '.join([f'AVG({p}) as {p}' for p in pollutants])
    if aggregation == 'Daily':
        group_col = 'DATE(datetime)'
        select_col = 'DATE(datetime) as date'
        order_col = 'date'
    elif aggregation == 'Weekly':
        group_col = 'STRFTIME("%Y-%W", datetime)'
        select_col = 'STRFTIME("%Y-%W", datetime) as week'
        order_col = 'week'
    elif aggregation == 'Monthly':
        group_col = 'STRFTIME("%Y-%m", datetime)'
        select_col = 'STRFTIME("%Y-%m", datetime) as month'
        order_col = 'month'
    else:
        group_col = 'datetime'
        select_col = 'datetime'
        order_col = 'datetime'
    sql = f"SELECT site, {select_col}, {poll_exprs} FROM {TABLE} WHERE date(datetime) >= ? AND date(datetime) <= ? AND site IN ({site_list}) GROUP BY site, {group_col} ORDER BY {order_col}"
    params = [start_date, end_date] + sites
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql(sql, conn, params=params)
    # Parse date column
    if aggregation == 'Daily':
        df['datetime'] = pd.to_datetime(df['date'])
    elif aggregation == 'Weekly':
        df['datetime'] = pd.to_datetime(df['week'] + '-1', format='%Y-%W-%w')
    elif aggregation == 'Monthly':
        df['datetime'] = pd.to_datetime(df['month'] + '-01')
    else:
        df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def downsample_df(df, time_col, value_cols, max_points=MAX_POINTS):
    if len(df) > max_points:
        freq = 'D' if (df[time_col].max() - df[time_col].min()).days < 60 else 'W'
        dfs = []
        for site, group in df.groupby('site'):
            group = group.set_index(time_col)
            resampled = group[value_cols].resample(freq).mean().reset_index()
            resampled['site'] = site
            dfs.append(resampled)
        df = pd.concat(dfs, ignore_index=True)
    return df

def main():
    st.set_page_config('Urban Air Quality Dashboard (SQLite)', layout='wide')
    st.title('🌫️ Urban Air Quality Dashboard (SQLite)')

    sites = get_sites()
    min_date, max_date = get_date_range()
    pollutant_cols = get_pollutants()

    # Default to most recent 30 days
    default_end = max_date
    default_start = max_date - datetime.timedelta(days=MAX_DAYS-1)

    with st.sidebar:
        st.header('Filters')
        date_range = st.date_input('Select Date Range', (default_start, default_end), min_value=min_date, max_value=max_date)
        selected_sites = st.multiselect('Select Sites (max 3)', sites, default=sites[:MAX_SITES], help='Selecting more than 3 sites will disable charts.')
        selected_pollutants = st.multiselect('Select Pollutants', pollutant_cols, default=pollutant_cols)
        aggregation = st.selectbox('Time Aggregation', ['Daily', 'Weekly', 'Monthly', 'Hourly'], index=0)
        heatmap_pollutant = st.selectbox('Pollutant for Heatmap', pollutant_cols, index=0)

    if len(selected_sites) > MAX_SITES:
        st.warning(f'Please select at most {MAX_SITES} sites. Charts are disabled.')
        return

    if not selected_sites or not selected_pollutants:
        st.warning('No data for selected filters.')
        return

    start_date = date_range[0].strftime('%Y-%m-%d')
    end_date = date_range[1].strftime('%Y-%m-%d')

    num_days = (date_range[1] - date_range[0]).days + 1
    if num_days > MAX_DAYS:
        st.warning(f'Please select a date range of at most {MAX_DAYS} days. Charts are disabled.')
        return

    # KPIs: Use filtered data only
    df_kpi = query_data(start_date, end_date, selected_sites, selected_pollutants)
    if df_kpi.empty:
        st.warning('No data for selected filters.')
        return
    col1, col2, col3, col4 = st.columns(4)
    avg_no2 = df_kpi['no2'].mean() if 'no2' in df_kpi.columns else np.nan
    avg_pm10 = df_kpi['pm10'].mean() if 'pm10' in df_kpi.columns else np.nan
    avg_pm25 = df_kpi['pm25'].mean() if 'pm25' in df_kpi.columns else np.nan
    poor_days = (df_kpi['pm25'] > 25).sum() if 'pm25' in df_kpi.columns else 0
    total_measurements = len(df_kpi[df_kpi['pm25'].notna()]) if 'pm25' in df_kpi.columns else 0
    percentage = (poor_days / total_measurements * 100) if total_measurements > 0 else 0
    col1.metric('🔵 Avg NO₂ (µg/m³)', f"{avg_no2:.1f}" if not np.isnan(avg_no2) else "N/A")
    col2.metric('🟡 Avg PM10 (µg/m³)', f"{avg_pm10:.1f}" if not np.isnan(avg_pm10) else "N/A")
    col3.metric('🔴 Avg PM2.5 (µg/m³)', f"{avg_pm25:.1f}" if not np.isnan(avg_pm25) else "N/A")
    col4.metric('⚠️ Poor Air Quality', f"{poor_days} readings", f"{percentage:.1f}% of total")

    # Time Series: Use SQL aggregation and downsample if needed
    df_agg = query_aggregated(start_date, end_date, selected_sites, selected_pollutants, aggregation)
    df_agg = downsample_df(df_agg, 'datetime', selected_pollutants)
    fig_ts = go.Figure()
    colors = {'no2': '#1f77b4', 'pm10': '#ff7f0e', 'pm25': '#d62728'}
    for pollutant in selected_pollutants:
        if pollutant in df_agg.columns:
            for site in selected_sites:
                df_plot = df_agg[df_agg['site'] == site].sort_values('datetime')
                threshold = WHO_THRESHOLDS.get(pollutant, None)
                if threshold is not None:
                    # Split into below and above threshold for line coloring
                    below = df_plot[pollutant] <= threshold
                    segments = []
                    current = []
                    current_color = below.iloc[0] if not below.empty else True
                    for i, is_below in enumerate(below):
                        if i == 0 or is_below == current_color:
                            current.append(i)
                        else:
                            segments.append((current_color, current))
                            current = [i]
                            current_color = is_below
                    if current:
                        segments.append((current_color, current))
                    # Plot each segment with appropriate color
                    for is_below, idxs in segments:
                        seg = df_plot.iloc[idxs]
                        fig_ts.add_trace(go.Scatter(
                            x=seg['datetime'],
                            y=seg[pollutant],
                            mode='lines+markers',
                            name=f'{pollutant.upper()} {site} (>{threshold})' if not is_below else f'{pollutant.upper()} {site} (≤{threshold})',
                            line=dict(color=('red' if not is_below else colors.get(pollutant, '#000000')), width=3),
                            marker=dict(size=4, color=('red' if not is_below else colors.get(pollutant, '#000000'))),
                            hovertemplate=f'<b>{pollutant.upper()} {site}</b><br>Date: %{{x}}<br>Value: %{{y:.1f}} µg/m³<extra></extra>',
                            text=None,
                            showlegend=True if idxs[0] == 0 else False
                        ))
                    # Add horizontal threshold line (once per pollutant)
                    fig_ts.add_hline(y=threshold, line_dash='dash', line_color='gray', annotation_text=f'{pollutant.upper()} WHO {threshold}', annotation_position='top left')
                else:
                    fig_ts.add_trace(go.Scatter(
                        x=df_plot['datetime'],
                        y=df_plot[pollutant],
                        mode='lines+markers',
                        name=f'{pollutant.upper()} {site}',
                        line=dict(color=colors.get(pollutant, '#000000'), width=3),
                        marker=dict(size=4),
                        hovertemplate=f'<b>{pollutant.upper()} {site}</b><br>Date: %{{x}}<br>Value: %{{y:.1f}} µg/m³<extra></extra>',
                        text=None
                    ))
    fig_ts.update_layout(
        title=f"📈 {aggregation} Air Quality Trends (line turns red = WHO exceedance)",
        xaxis_title="Date",
        yaxis_title="Concentration (µg/m³)",
        hovermode='x unified',
        height=500,
        showlegend=True,
        template="plotly_white"
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Site Comparison: Show top N sites by default, allow toggle for all
    df_site = df_agg.groupby('site')[selected_pollutants].mean().reset_index()
    N = 15
    df_melted = df_site.melt(
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
        title=f"📊 Average Pollutant Levels by Site",
        labels={'Concentration': 'Concentration (µg/m³)'},
        template="plotly_white",
        height=500
    )
    fig_site.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_site, use_container_width=True)

    # Correlation: Use filtered data
    from plotly.subplots import make_subplots
    fig_corr = make_subplots(
        rows=1, cols=3,
        subplot_titles=("PM2.5 vs PM10", "PM2.5 vs NO₂", "PM10 vs NO₂"),
        horizontal_spacing=0.1
    )
    if 'pm25' in df_kpi.columns and 'pm10' in df_kpi.columns:
        valid_data = df_kpi[['pm25', 'pm10']].dropna()
        if not valid_data.empty:
            fig_corr.add_trace(
                go.Scatter(x=valid_data['pm10'], y=valid_data['pm25'], mode='markers',
                          name='PM2.5 vs PM10', marker=dict(color='blue', opacity=0.6)),
                row=1, col=1
            )
    if 'pm25' in df_kpi.columns and 'no2' in df_kpi.columns:
        valid_data = df_kpi[['pm25', 'no2']].dropna()
        if not valid_data.empty:
            fig_corr.add_trace(
                go.Scatter(x=valid_data['no2'], y=valid_data['pm25'], mode='markers',
                          name='PM2.5 vs NO₂', marker=dict(color='red', opacity=0.6)),
                row=1, col=2
            )
    if 'pm10' in df_kpi.columns and 'no2' in df_kpi.columns:
        valid_data = df_kpi[['pm10', 'no2']].dropna()
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
    st.plotly_chart(fig_corr, use_container_width=True)

    # Data Export
    st.download_button(
        label="📥 Download CSV",
        data=df_kpi.to_csv(index=False),
        file_name=f"air_quality_filtered_{datetime.date.today()}.csv",
        mime="text/csv"
    )

    with st.expander('💡 Air Quality Facts & Information', expanded=True):
        st.markdown('''
        - 🌍 **WHO Air Quality Guidelines:**
            - PM2.5: Annual mean ≤ 15 µg/m³, 24-hour mean ≤ 45 µg/m³
            - PM10: Annual mean ≤ 45 µg/m³, 24-hour mean ≤ 50 µg/m³
            - NO₂: Annual mean ≤ 10 µg/m³, 24-hour mean ≤ 25 µg/m³
        - 🔬 **What are these pollutants?**
            - PM2.5: Fine particulate matter ≤ 2.5 micrometers
            - PM10: Coarse particulate matter ≤ 10 micrometers
            - NO₂: Nitrogen dioxide, mainly from vehicles/industry
        - ⚕️ **Health Impacts:**
            - Poor air quality can cause respiratory/cardiovascular issues
            - Children, elderly, and those with health conditions are most vulnerable
            - Long-term exposure reduces life expectancy
        ''')

if __name__ == '__main__':
    main() 