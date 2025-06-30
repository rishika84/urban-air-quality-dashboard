import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="🌫️ Urban Air Quality Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f8f5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2E8B57;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the air quality data"""
    try:
        # Read the CSV file
        df = pd.read_csv('combined_cleaned_air_quality.csv')
        
        # Combine Date and Time into Datetime column
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d-%m-%Y %H:%M:%S')
        
        # Convert pollutant columns to numeric, handling missing values
        pollutant_cols = ['Nitrogen dioxide', 'PM10', 'PM2.5']
        for col in pollutant_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by datetime
        df = df.sort_values('Datetime').reset_index(drop=True)
        
        return df, pollutant_cols
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def filter_data(df, date_range, selected_sites, selected_pollutants):
    """Filter data based on user selections"""
    # Filter by date range
    mask = (df['Datetime'].dt.date >= date_range[0]) & (df['Datetime'].dt.date <= date_range[1])
    filtered_df = df[mask]
    
    # Filter by sites
    if selected_sites:
        filtered_df = filtered_df[filtered_df['Site Name'].isin(selected_sites)]
    
    return filtered_df

def create_kpi_metrics(df, pollutants):
    """Create KPI metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if not df.empty:
            avg_no2 = df['Nitrogen dioxide'].mean()
            st.metric(
                label="🔵 Avg NO₂ (µg/m³)",
                value=f"{avg_no2:.1f}" if not np.isnan(avg_no2) else "N/A"
            )
        
    with col2:
        if not df.empty:
            avg_pm10 = df['PM10'].mean()
            st.metric(
                label="🟡 Avg PM10 (µg/m³)",
                value=f"{avg_pm10:.1f}" if not np.isnan(avg_pm10) else "N/A"
            )
    
    with col3:
        if not df.empty:
            avg_pm25 = df['PM2.5'].mean()
            st.metric(
                label="🔴 Avg PM2.5 (µg/m³)",
                value=f"{avg_pm25:.1f}" if not np.isnan(avg_pm25) else "N/A"
            )
    
    with col4:
        if not df.empty:
            # Count poor air quality days (PM2.5 > 25)
            poor_days = (df['PM2.5'] > 25).sum()
            total_measurements = len(df[df['PM2.5'].notna()])
            percentage = (poor_days / total_measurements * 100) if total_measurements > 0 else 0
            st.metric(
                label="⚠️ Poor Air Quality",
                value=f"{poor_days} readings",
                delta=f"{percentage:.1f}% of total"
            )

def create_time_series_chart(df, selected_pollutants, aggregation):
    """Create time series line chart"""
    if df.empty or not selected_pollutants:
        st.warning("No data available for the selected filters.")
        return
    
    # Aggregate data based on selection
    df_agg = df.copy()
    
    if aggregation == "Daily":
        df_agg = df_agg.groupby([df_agg['Datetime'].dt.date, 'Site Name'])[selected_pollutants].mean().reset_index()
        df_agg['Datetime'] = pd.to_datetime(df_agg['Datetime'])
    elif aggregation == "Weekly":
        df_agg['Week'] = df_agg['Datetime'].dt.to_period('W')
        df_agg = df_agg.groupby(['Week', 'Site Name'])[selected_pollutants].mean().reset_index()
        df_agg['Datetime'] = df_agg['Week'].dt.start_time
    elif aggregation == "Monthly":
        df_agg['Month'] = df_agg['Datetime'].dt.to_period('M')
        df_agg = df_agg.groupby(['Month', 'Site Name'])[selected_pollutants].mean().reset_index()
        df_agg['Datetime'] = df_agg['Month'].dt.start_time
    
    # Create plotly figure
    fig = go.Figure()
    
    colors = {'Nitrogen dioxide': '#1f77b4', 'PM10': '#ff7f0e', 'PM2.5': '#d62728'}
    
    for pollutant in selected_pollutants:
        if pollutant in df_agg.columns:
            # Average across all sites for cleaner visualization
            df_plot = df_agg.groupby('Datetime')[pollutant].mean().reset_index()
            
            fig.add_trace(go.Scatter(
                x=df_plot['Datetime'],
                y=df_plot[pollutant],
                mode='lines+markers',
                name=pollutant,
                line=dict(color=colors.get(pollutant, '#000000'), width=3),
                marker=dict(size=6),
                hovertemplate=f'<b>{pollutant}</b><br>Date: %{{x}}<br>Value: %{{y:.1f}} µg/m³<extra></extra>'
            ))
    
    fig.update_layout(
        title=f"📈 {aggregation} Air Quality Trends",
        xaxis_title="Date",
        yaxis_title="Concentration (µg/m³)",
        hovermode='x unified',
        height=500,
        showlegend=True,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_site_comparison_chart(df, selected_pollutants):
    """Create bar chart comparing sites"""
    if df.empty or not selected_pollutants:
        st.warning("No data available for the selected filters.")
        return
    
    # Calculate average by site
    site_averages = df.groupby('Site Name')[selected_pollutants].mean().reset_index()
    
    # Melt for plotting
    df_melted = site_averages.melt(
        id_vars='Site Name', 
        value_vars=selected_pollutants,
        var_name='Pollutant', 
        value_name='Concentration'
    )
    
    fig = px.bar(
        df_melted,
        x='Site Name',
        y='Concentration',
        color='Pollutant',
        title="📊 Average Pollutant Levels by Site",
        labels={'Concentration': 'Concentration (µg/m³)'},
        template="plotly_white",
        height=500
    )
    
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def create_hourly_heatmap(df, pollutant):
    """Create hourly heatmap"""
    if df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Create hour and day of week columns
    df_heat = df.copy()
    df_heat['Hour'] = df_heat['Datetime'].dt.hour
    df_heat['DayOfWeek'] = df_heat['Datetime'].dt.day_name()
    
    # Calculate average by hour and day of week
    heatmap_data = df_heat.groupby(['DayOfWeek', 'Hour'])[pollutant].mean().reset_index()
    
    # Pivot for heatmap
    pivot_data = heatmap_data.pivot(index='DayOfWeek', columns='Hour', values=pollutant)
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_data = pivot_data.reindex(day_order)
    
    fig = px.imshow(
        pivot_data,
        title=f"🕐 Hourly {pollutant} Concentration Heatmap",
        labels=dict(x="Hour of Day", y="Day of Week", color=f"{pollutant} (µg/m³)"),
        aspect="auto",
        color_continuous_scale="Reds",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_correlation_scatter(df):
    """Create scatter plot for pollutant correlations"""
    if df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Create subplots for different correlations
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("PM2.5 vs PM10", "PM2.5 vs NO₂", "PM10 vs NO₂"),
        horizontal_spacing=0.1
    )
    
    # PM2.5 vs PM10
    valid_data = df[['PM2.5', 'PM10']].dropna()
    if not valid_data.empty:
        fig.add_trace(
            go.Scatter(x=valid_data['PM10'], y=valid_data['PM2.5'], mode='markers',
                      name='PM2.5 vs PM10', marker=dict(color='blue', opacity=0.6)),
            row=1, col=1
        )
    
    # PM2.5 vs NO₂
    valid_data = df[['PM2.5', 'Nitrogen dioxide']].dropna()
    if not valid_data.empty:
        fig.add_trace(
            go.Scatter(x=valid_data['Nitrogen dioxide'], y=valid_data['PM2.5'], mode='markers',
                      name='PM2.5 vs NO₂', marker=dict(color='red', opacity=0.6)),
            row=1, col=2
        )
    
    # PM10 vs NO₂
    valid_data = df[['PM10', 'Nitrogen dioxide']].dropna()
    if not valid_data.empty:
        fig.add_trace(
            go.Scatter(x=valid_data['Nitrogen dioxide'], y=valid_data['PM10'], mode='markers',
                      name='PM10 vs NO₂', marker=dict(color='green', opacity=0.6)),
            row=1, col=3
        )
    
    fig.update_layout(
        title="🔗 Pollutant Correlation Analysis",
        height=500,
        showlegend=False,
        template="plotly_white"
    )
    
    fig.update_xaxes(title_text="PM10 (µg/m³)", row=1, col=1)
    fig.update_xaxes(title_text="NO₂ (µg/m³)", row=1, col=2)
    fig.update_xaxes(title_text="NO₂ (µg/m³)", row=1, col=3)
    fig.update_yaxes(title_text="PM2.5 (µg/m³)", row=1, col=1)
    fig.update_yaxes(title_text="PM2.5 (µg/m³)", row=1, col=2)
    fig.update_yaxes(title_text="PM10 (µg/m³)", row=1, col=3)
    
    st.plotly_chart(fig, use_container_width=True)

def check_air_quality_alerts(df):
    """Check for air quality alerts"""
    if df.empty:
        return
    
    alerts = []
    
    # WHO guidelines: PM2.5 > 25 µg/m³, PM10 > 50 µg/m³, NO₂ > 200 µg/m³
    pm25_exceed = (df['PM2.5'] > 25).sum()
    pm10_exceed = (df['PM10'] > 50).sum()
    no2_exceed = (df['Nitrogen dioxide'] > 200).sum()
    
    if pm25_exceed > 0:
        alerts.append(f"⚠️ **PM2.5 Alert**: {pm25_exceed} readings exceed WHO guidelines (>25 µg/m³)")
    
    if pm10_exceed > 0:
        alerts.append(f"⚠️ **PM10 Alert**: {pm10_exceed} readings exceed WHO guidelines (>50 µg/m³)")
    
    if no2_exceed > 0:
        alerts.append(f"⚠️ **NO₂ Alert**: {no2_exceed} readings exceed WHO guidelines (>200 µg/m³)")
    
    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.success("✅ All readings within WHO air quality guidelines for the selected period!")

def main():
    # Header
    st.markdown('<h1 class="main-header">🌫️ Urban Air Quality Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    df, pollutant_cols = load_and_preprocess_data()
    
    if df is None:
        st.error("Failed to load data. Please ensure 'combined_cleaned_air_quality.csv' is in the correct location.")
        return
    
    # Sidebar
    st.sidebar.markdown('<p class="sidebar-header">🧭 Dashboard Controls</p>', unsafe_allow_html=True)
    
    # Date range selector
    min_date = df['Datetime'].dt.date.min()
    max_date = df['Datetime'].dt.date.max()
    
    date_range = st.sidebar.date_input(
        "📅 Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Site selector
    sites = sorted(df['Site Name'].unique())
    selected_sites = st.sidebar.multiselect(
        "📍 Select Sites",
        options=sites,
        default=sites[:5] if len(sites) > 5 else sites
    )
    
    # Pollutant selector
    selected_pollutants = st.sidebar.multiselect(
        "🔬 Select Pollutants",
        options=pollutant_cols,
        default=pollutant_cols
    )
    
    # Time aggregation
    aggregation = st.sidebar.selectbox(
        "📊 Time Aggregation",
        options=["Hourly", "Daily", "Weekly", "Monthly"],
        index=1
    )
    
    # Filter data
    if len(date_range) == 2:
        filtered_df = filter_data(df, date_range, selected_sites, selected_pollutants)
    else:
        filtered_df = df.copy()
    
    # Main dashboard
    if not filtered_df.empty:
        # KPI Metrics
        st.subheader("📊 Key Performance Indicators")
        create_kpi_metrics(filtered_df, selected_pollutants)
        
        st.markdown("---")
        
        # Air Quality Alerts
        st.subheader("🚨 Air Quality Status")
        check_air_quality_alerts(filtered_df)
        
        st.markdown("---")
        
        # Time Series Chart
        st.subheader("📈 Time Series Analysis")
        create_time_series_chart(filtered_df, selected_pollutants, aggregation)
        
        # Two column layout for other charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Site Comparison")
            create_site_comparison_chart(filtered_df, selected_pollutants)
        
        with col2:
            st.subheader("🕐 Hourly Patterns")
            heatmap_pollutant = st.selectbox(
                "Select pollutant for heatmap:",
                selected_pollutants,
                key="heatmap_pollutant"
            )
            if heatmap_pollutant:
                create_hourly_heatmap(filtered_df, heatmap_pollutant)
        
        # Correlation Analysis
        st.subheader("🔗 Pollutant Correlations")
        create_correlation_scatter(filtered_df)
        
        # Data Export
        st.markdown("---")
        st.subheader("📤 Data Export")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write(f"**Filtered Dataset**: {len(filtered_df):,} records from {len(filtered_df['Site Name'].unique())} sites")
        
        with col2:
            # Create download button
            csv_buffer = BytesIO()
            filtered_df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv_buffer.getvalue(),
                file_name=f"air_quality_filtered_{datetime.date.today()}.csv",
                mime="text/csv"
            )
    
    else:
        st.warning("No data available for the selected filters. Please adjust your selection.")
    
    # Air Quality Facts
    st.markdown("---")
    with st.expander("💡 Air Quality Facts & Information"):
        st.markdown("""
        **🌍 WHO Air Quality Guidelines:**
        - **PM2.5**: Annual mean should not exceed 15 µg/m³, 24-hour mean should not exceed 45 µg/m³
        - **PM10**: Annual mean should not exceed 45 µg/m³, 24-hour mean should not exceed 50 µg/m³
        - **NO₂**: Annual mean should not exceed 10 µg/m³, 24-hour mean should not exceed 25 µg/m³
        
        **🔬 What are these pollutants?**
        - **PM2.5**: Fine particulate matter with diameter ≤ 2.5 micrometers
        - **PM10**: Coarse particulate matter with diameter ≤ 10 micrometers  
        - **NO₂**: Nitrogen dioxide, primarily from vehicle emissions and industrial processes
        
        **⚕️ Health Impacts:**
        - Poor air quality can cause respiratory problems, cardiovascular disease, and premature death
        - Children, elderly, and people with existing health conditions are most vulnerable
        - Long-term exposure to air pollution reduces life expectancy
        """)

if __name__ == "__main__":
    main()