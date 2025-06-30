import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sqlite3
import datetime

DB_FILE = 'air_quality.sqlite'
TABLE = 'air_quality'
WHO_GUIDELINES = {'pm25': 15, 'pm10': 45, 'no2': 10}
STATUS_BANDS = {
    'pm25': [(0, 15, 'Good', '#43a047'), (15, 35, 'Moderate', '#fbc02d'), (35, 55, 'Poor', '#e64a19'), (55, 150, 'Unhealthy', '#d32f2f'), (150, 1000, 'Hazardous', '#6d4c41')],
    'pm10': [(0, 45, 'Good', '#43a047'), (45, 75, 'Moderate', '#fbc02d'), (75, 150, 'Poor', '#e64a19'), (150, 350, 'Unhealthy', '#d32f2f'), (350, 1000, 'Hazardous', '#6d4c41')],
    'no2': [(0, 10, 'Good', '#43a047'), (10, 25, 'Moderate', '#fbc02d'), (25, 50, 'Poor', '#e64a19'), (50, 200, 'Unhealthy', '#d32f2f'), (200, 1000, 'Hazardous', '#6d4c41')],
}
POLLUTANT_EXPLAIN = {
    'pm25': 'PM2.5 are fine inhalable particles, with diameters that are generally 2.5 micrometers and smaller. They can penetrate deep into the lungs and even enter the bloodstream.',
    'pm10': 'PM10 are inhalable particles, with diameters that are generally 10 micrometers and smaller. They can cause respiratory issues and aggravate heart diseases.',
    'no2': 'NO₂ (Nitrogen Dioxide) is a gas mainly produced from burning fuel. High levels can irritate airways and worsen asthma.',
}

PM25_SOURCES = [
    ("🌬️", "Windblown Dust", "Daily activities like construction or other practices"),
    ("🏠", "Home-related emission", "Household activities, such as cooking and heating"),
    ("🏭", "Factories and industries' emission", "Regular operations in factories and industries"),
    ("⚡", "Power plants generation", "Emission from Routine energy production in power plants"),
    ("🔥", "Landfill fires", "Fires in landfills, often caused by waste mismanagement"),
    ("🚚", "Transportation emission", "Diesel operated Daily vehicles produces exhaust"),
    ("🌾", "Human-caused emissions", "Common practices like open burning of waste or agricultural residues"),
]
PM10_SOURCES = [
    ("🌬️", "Wind-blown dust", "Dust lifted and spread by the wind from bare soil."),
    ("🏗️", "Construction sites", "Dust and pollutants from building activities."),
    ("🏭", "Industries", "Releases various pollutants from different processes."),
    ("🔥", "Waste burning", "Smoke and toxins from burning waste materials."),
    ("🏞️", "Landfills", "Emissions from decomposing waste in big landfills."),
    ("🚗", "Vehicles exhausts", "Emissions of harmful gases and particles from cars."),
]
NO2_SOURCES = [
    ("💥", "Explosives and welding", "Release during high-temperature processes."),
    ("💡", "Lighting", "Emissions from certain types of lighting."),
    ("⚡", "Power-generating plants", "Emissions from fossil fuels burning for electricity."),
    ("🚦", "Road traffic", "Overall contributions from vehicles on the road."),
    ("🚗", "Motor vehicles", "Exhaust gases from cars and trucks."),
]

PM25_IMPACTS = [
    ("Fatigue", "Feeling unusually tired or weak."),
    ("Irritation in Eyes", "Redness, itching, and discomfort in your eyes."),
    ("Headaches", "Frequent or intense headaches."),
    ("Aggravated asthma", "Increased asthma attacks and symptoms."),
    ("Breathing problems", "Coughing, wheezing, and shortness of breath."),
]
PM10_IMPACTS = [
    ("Eyes irritations", "Redness and discomfort in the eyes."),
    ("Allergies", "Airborne allergens causing sneezing and itching."),
    ("Cough and runny nose", "Respiratory irritation cause to coughing."),
    ("Chest tightness", "Constriction feeling in the chest."),
    ("Breathing difficulty", "Trouble in breathing or discomfort in the chest."),
]
NO2_IMPACTS = [
    ("Difficulty in breathing", "Shortness of breath and reduced lung function."),
    ("Coughing", "Regular cough due to airways irritations."),
    ("Wheezing", "More whistling sound while breathing."),
    ("Reduce smelling ability", "Decreased sense of smell due to irritation."),
]

@st.cache_data(show_spinner=False)
def get_sites():
    with sqlite3.connect(DB_FILE) as conn:
        sites = pd.read_sql(f"SELECT DISTINCT site FROM {TABLE} ORDER BY site", conn)['site'].tolist()
    return sites

@st.cache_data(show_spinner=False)
def get_date_range():
    with sqlite3.connect(DB_FILE) as conn:
        min_date = pd.read_sql(f"SELECT MIN(datetime) as min_date FROM {TABLE}", conn)['min_date'].iloc[0]
        max_date = pd.read_sql(f"SELECT MAX(datetime) as max_date FROM {TABLE}", conn)['max_date'].iloc[0]
    min_date = pd.to_datetime(min_date).date()
    max_date = pd.to_datetime(max_date).date()
    # Clamp min_date to 2024-01-01 if older
    min_date = max(min_date, datetime.date(2024, 1, 1))
    return min_date, max_date

@st.cache_data(show_spinner=False)
def get_weekly_data(pollutant, selected_sites, start_date, end_date):
    # Convert to string for SQL
    start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    query = f"SELECT datetime, site, {pollutant} FROM {TABLE} WHERE datetime >= ? AND datetime <= ?"
    params = [start_date, end_date]
    if selected_sites:
        placeholders = ','.join(['?']*len(selected_sites))
        query += f" AND site IN ({placeholders})"
        params.extend(selected_sites)
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql(query, conn, params=params, parse_dates=['datetime'])
    df = df.dropna(subset=[pollutant])
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df['week'] = df['datetime'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly = df.groupby('week')[pollutant].agg(['mean', 'min', 'max', 'count'])
    weekly = weekly.reset_index()
    return weekly, df

def get_status(value, pollutant):
    for low, high, label, color in STATUS_BANDS[pollutant]:
        if low <= value < high:
            return label, color
    return 'Unknown', '#757575'

def kpi_card(label, value, icon, color, helptext=None, delta=None):
    # Smaller, compact card with .kpi-card class for shadow and animation
    return f'<div class="kpi-card"><span style="font-size:1.4rem;">{icon}</span>' \
           f'<div>' \
           f'<div style="font-size:0.95rem;font-weight:600;color:#555;">{label}</div>' \
           f'<div style="font-size:1.4rem;font-weight:bold;color:{color}">{value}</div>' \
           f'{f"<div style=\"font-size:0.8rem;color:#888;\">{helptext}</div>" if helptext else ""}' \
           f'{f"<div style=\"font-size:0.9rem;color:#388e3c;\">{delta}</div>" if delta else ""}' \
           f'</div></div>'

def weekly_line_chart(weekly, pollutant, guideline):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly['week'], y=weekly['mean'], mode='lines+markers', name='Weekly Avg', line=dict(color='#1976d2', width=3)))
    fig.add_trace(go.Scatter(x=weekly['week'], y=weekly['max'], mode='lines', name='Weekly Max', line=dict(color='#d32f2f', dash='dot')))
    fig.add_trace(go.Scatter(x=weekly['week'], y=weekly['min'], mode='lines', name='Weekly Min', line=dict(color='#43a047', dash='dot')))
    # Threshold line
    fig.add_trace(go.Scatter(x=weekly['week'], y=[guideline]*len(weekly), mode='lines', name='WHO Guideline', line=dict(color='#fbc02d', width=2, dash='dash')))
    # Highlight exceedances
    exceed = weekly['mean'] > guideline
    fig.add_trace(go.Scatter(x=weekly['week'][exceed], y=weekly['mean'][exceed], mode='markers', name='Exceedance', marker=dict(color='#d32f2f', size=12, symbol='circle-open')))
    fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0), yaxis_title=f'{pollutant.upper()} (µg/m³)', xaxis_title='Week', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    return fig

def sources_cards(sources):
    cards = []
    for icon, title, desc in sources:
        cards.append(f'<div style="flex:1 1 220px;min-width:180px;max-width:260px;background:#222c2f0d;border-radius:1rem;padding:1.1rem 1.2rem;margin:0.5rem;box-shadow:0 1px 4px #0001;display:flex;flex-direction:column;align-items:flex-start;"><span style="font-size:1.6rem;">{icon}</span><div style="font-weight:700;font-size:1.1rem;margin-top:0.5rem;">{title}</div><div style="font-size:0.98rem;margin-top:0.2rem;opacity:0.85;">{desc}</div></div>')
    # No triple backticks, no markdown code block, just HTML string
    return '<div style="display:flex;flex-wrap:wrap;justify-content:center;gap:0.5rem;">' + ''.join(cards) + '</div>'

def impacts_cards(title, impacts):
    # Add shadow and hover animation to impact cards
    style = '''
    <style>
    .impact-card-row {display:flex;flex-wrap:wrap;justify-content:center;gap:0.5rem;}
    .impact-card {
        flex:1 1 220px;min-width:180px;max-width:260px;background:#f8f9fb;
        border-radius:1rem;padding:1.1rem 1.2rem;margin:0.5rem;
        box-shadow:0 2px 12px #0002, 0 1.5px 4px #0001;
        display:flex;flex-direction:column;align-items:flex-start;
        transition:transform 0.18s cubic-bezier(.4,1.5,.5,1),box-shadow 0.18s;
    }
    .impact-card:hover {
        transform:scale(1.06) translateY(-4px);
        box-shadow:0 8px 24px #0003, 0 2px 8px #0002;
    }
    </style>
    '''
    cards = []
    for impact, desc in impacts:
        cards.append(f'<div class="impact-card"><div style="font-weight:700;font-size:1.08rem;margin-bottom:0.2rem;">{impact}</div><div style="font-size:0.98rem;opacity:0.85;">{desc}</div></div>')
    return style + f'''<div style="background:#f8f9fb;border-radius:1.5rem;padding:2.2rem 1.5rem 1.5rem 1.5rem;margin-top:2.2rem;margin-bottom:1.2rem;">
    <div style="font-size:1.6rem;font-weight:700;margin-bottom:1.2rem;">{title}</div>
    <div class="impact-card-row">{''.join(cards)}</div>
    </div>'''

def main():
    st.set_page_config('Urban Air Quality Dashboard', layout='wide')
    st.markdown('''
    <style>
    div.block-container{padding-top:1rem;}
    .kpi-row{display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1.2rem;}
    .kpi-card{
        display:flex;align-items:center;gap:0.5rem;background:#fff;
        border-radius:0.8rem;padding:0.7rem 1.1rem;
        box-shadow:0 2px 12px #0002, 0 1.5px 4px #0001;
        min-width:120px;max-width:180px;transition:transform 0.18s cubic-bezier(.4,1.5,.5,1),box-shadow 0.18s;
    }
    .kpi-card:hover{
        transform:scale(1.06) translateY(-4px);
        box-shadow:0 8px 24px #0003, 0 2px 8px #0002;
    }
    .explain-card{background:#f1f8e9;border-radius:16px;padding:1.5rem;margin-top:1rem;}
    </style>
    ''', unsafe_allow_html=True)

    tab_labels = ["PM2.5", "PM10", "NO₂"]
    pollutant_keys = ['pm25', 'pm10', 'no2']

    # Use a hidden radio to track the selected tab index
    selected_tab_idx = st.radio("", tab_labels, horizontal=True, label_visibility="collapsed")
    selected_tab_idx = tab_labels.index(selected_tab_idx)

    with st.sidebar:
        st.header('Dashboard Controls')
        sites = get_sites()
        min_date, max_date = get_date_range()
        default_site = ['Birmingham'] if 'Birmingham' in sites else sites[:1]
        selected_sites = st.multiselect('Select Sites', sites, default=default_site, max_selections=3)
        date_range = st.date_input('Select Date Range', [min_date, max_date], min_value=min_date, max_value=max_date)
        st.write(f"**Total Sites:** {len(selected_sites)}")
        # Info card for selected tab
        st.markdown(f'<div class="explain-card"><b>What is {pollutant_keys[selected_tab_idx].upper()}?</b><br>{POLLUTANT_EXPLAIN[pollutant_keys[selected_tab_idx]]}</div>', unsafe_allow_html=True)

    # Only show the content for the selected tab
    i = selected_tab_idx
    pollutant = pollutant_keys[i]
    st.title(f"Urban Air Quality Dashboard – {pollutant.upper()}")
    try:
        with st.spinner('Loading data...'):
            dr = tuple(date_range)
            weekly, df = get_weekly_data(
                pollutant,
                selected_sites,
                pd.to_datetime(dr[0]),
                pd.to_datetime(dr[1])
            )
        if weekly.empty:
            st.warning('No data available for this pollutant and selection.')
        else:
            avg_val = df[pollutant].mean()
            min_val = df[pollutant].min()
            max_val = df[pollutant].max()
            guideline = WHO_GUIDELINES[pollutant]
            exceed_count = (weekly['mean'] > guideline).sum()
            # KPI cards
            kpi_html = f'<div class="kpi-row">' + \
                kpi_card("Average", f"{avg_val:.1f} µg/m³", "📊", "#388e3c") + \
                kpi_card("Minimum", f"{min_val:.1f} µg/m³", "📉", "#1976d2") + \
                kpi_card("Maximum", f"{max_val:.1f} µg/m³", "📈", "#d32f2f") + \
                kpi_card("WHO Guideline", f"{guideline} µg/m³", "✅", "#fbc02d") + \
                kpi_card("Exceedance Weeks", f"{exceed_count}", "⚠️", "#d32f2f", f"> {guideline} µg/m³") + \
                '</div>'
            st.markdown(kpi_html, unsafe_allow_html=True)
            # Weekly line chart
            st.plotly_chart(weekly_line_chart(weekly, pollutant, guideline), use_container_width=True, config={"displayModeBar": False})
            # Sources section below the chart
            st.markdown('<h4 style="margin-top:2.5rem;margin-bottom:0.7rem;">Sources of ' + pollutant.upper() + '</h4>', unsafe_allow_html=True)
            if pollutant == 'pm25':
                st.markdown(sources_cards(PM25_SOURCES), unsafe_allow_html=True)
            elif pollutant == 'pm10':
                st.markdown(sources_cards(PM10_SOURCES), unsafe_allow_html=True)
            elif pollutant == 'no2':
                st.markdown(sources_cards(NO2_SOURCES), unsafe_allow_html=True)
            # Impacts section below the sources
            if pollutant == 'pm25':
                st.markdown(impacts_cards('Short-Term PM2.5 Exposure Impacts', PM25_IMPACTS), unsafe_allow_html=True)
            elif pollutant == 'pm10':
                st.markdown(impacts_cards('Short-Term PM10 Exposure Impacts', PM10_IMPACTS), unsafe_allow_html=True)
            elif pollutant == 'no2':
                st.markdown(impacts_cards('Short-Term NO2 Exposure Impacts', NO2_IMPACTS), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading data: {e}")

    st.caption('Made with ❤️ for clean air. Inspired by AQI.com and modern dashboards. Powered by Streamlit.')

if __name__ == '__main__':
    main() 