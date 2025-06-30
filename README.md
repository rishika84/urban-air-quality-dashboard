# Urban Air Quality Dashboard

A modern, interactive air quality monitoring dashboard built with Python, Streamlit, and Plotly. This dashboard provides real-time visualization and analysis of air quality data from multiple monitoring sites across urban areas.

## 🌟 Features

### Interactive Dashboard
- **Multi-pollutant Analysis**: PM2.5, PM10, and NO₂ monitoring with dedicated tabs
- **Real-time KPIs**: Average, Min, Max values with WHO guideline comparisons
- **Weekly Trend Visualization**: Interactive line charts with exceedance markers
- **Site Comparison**: Multi-site selection (up to 3 sites simultaneously)
- **Date Range Filtering**: Flexible date selection for historical analysis

### Performance Optimized
- **Smart Caching**: Efficient data loading with Streamlit caching
- **SQL Queries**: Fast data filtering using SQLite/Parquet databases
- **Downsampling**: Optimized for large datasets
- **Responsive Design**: Works seamlessly on desktop and mobile

### Modern UI/UX
- **Professional Design**: Inspired by AQI.com and modern dashboards
- **Interactive Cards**: Hover animations and visual feedback
- **Status Indicators**: Color-coded air quality status bars
- **Information Cards**: Sources and health impact information for each pollutant

## 📊 Data Sources

The dashboard supports multiple data formats:
- **CSV Files**: Raw air quality monitoring data
- **SQLite Database**: Optimized for fast queries
- **Parquet Files**: Columnar storage for efficient data access
- **DuckDB**: High-performance analytical database

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/urban-air-quality-dashboard.git
   cd urban-air-quality-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**
   - Place your CSV files in the project directory
   - Or use the provided data conversion scripts to create SQLite/Parquet databases

4. **Run the dashboard**
   ```bash
   streamlit run dashboard_modern.py
   ```

## 📁 Project Structure

```
CUrbanAirQuality/
├── dashboard_modern.py          # Main dashboard application
├── dashboard_dash_parquet.py    # Dash-based dashboard (Parquet)
├── dashboard_dash_sqlite.py     # Dash-based dashboard (SQLite)
├── dashboard_parquet.py         # Streamlit dashboard (Parquet)
├── dashboard_sqlite.py          # Streamlit dashboard (SQLite)
├── clean_defra_data.py          # Data cleaning script
├── convert_to_parquet.py        # CSV to Parquet converter
├── convert_to_sqlite.py         # CSV to SQLite converter
├── convert_to_duckdb.py         # CSV to DuckDB converter
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── *.csv                        # Air quality data files
```

## 🎯 Usage

### Starting the Dashboard
1. Open your terminal/command prompt
2. Navigate to the project directory
3. Run: `streamlit run dashboard_modern.py`
4. Open your browser to the provided URL (usually http://localhost:8501)

### Using the Dashboard
1. **Select Sites**: Choose up to 3 monitoring sites from the sidebar
2. **Set Date Range**: Select your desired time period (2024-2025)
3. **Explore Pollutants**: Switch between PM2.5, PM10, and NO₂ tabs
4. **Analyze Trends**: View weekly trends and exceedance patterns
5. **Download Data**: Use the Data Explorer tab to export filtered data

### Key Features
- **KPI Cards**: View average, min, max values and WHO guideline comparisons
- **Trend Charts**: Interactive weekly line charts with threshold indicators
- **Status Bars**: Visual air quality status with color-coded segments
- **Information Cards**: Learn about pollutant sources and health impacts

## 🔧 Configuration

### Data Sources
The dashboard automatically detects available data sources:
- CSV files: `*.csv`
- SQLite database: `air_quality.sqlite`
- Parquet files: `*.parquet`
- DuckDB database: `air_quality.duckdb`

### Customization
- Modify `dashboard_modern.py` to change the dashboard layout
- Update color schemes in the CSS styles
- Add new pollutants by extending the pollutant definitions

## 📈 Performance Tips

1. **Use Parquet/SQLite**: Convert large CSV files to Parquet or SQLite for better performance
2. **Limit Date Ranges**: Select smaller date ranges for faster loading
3. **Reduce Site Selection**: Fewer selected sites mean faster queries
4. **Enable Caching**: The dashboard uses Streamlit caching for optimal performance

## 🛠️ Development

### Adding New Features
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Data Processing
- Use `clean_defra_data.py` to clean raw air quality data
- Use conversion scripts to optimize data storage
- Ensure data format consistency across all sources

## 📊 Data Format

The dashboard expects air quality data with the following columns:
- `Date`: Date of measurement
- `Site`: Monitoring site identifier
- `PM2.5`: Particulate matter 2.5 (μg/m³)
- `PM10`: Particulate matter 10 (μg/m³)
- `NO2`: Nitrogen dioxide (μg/m³)

## 🌐 Deployment

### Local Deployment
```bash
streamlit run dashboard_modern.py
```

### Cloud Deployment
The dashboard can be deployed to:
- **Streamlit Cloud**: Free hosting for Streamlit apps
- **Heroku**: Cloud platform deployment
- **AWS/GCP/Azure**: Cloud infrastructure deployment

### Streamlit Cloud Deployment
1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. Deploy automatically with each push

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/yourusername/urban-air-quality-dashboard/issues) page
2. Create a new issue with detailed information
3. Include error messages and system information

## 🔄 Version History

- **v1.0.0**: Initial release with modern dashboard design
- **v1.1.0**: Added performance optimizations and caching
- **v1.2.0**: Enhanced UI with hover animations and improved styling

---

**Built with ❤️ using Streamlit, Plotly, and Python** 