import pandas as pd
import os
import numpy as np

# Step 1: Set the correct folder path (your Downloads subfolder)
folder_path = "C:/Users/aloor/Downloads/CUrbanAirQuality"  # Updated to your folder name

# Step 2: List all CSV files
files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
print(f"Found {len(files)} CSV files")

# Step 3: Common pollutant column names (case-insensitive)
pollutants = ['pm2.5', 'pm10', 'no2', 'o3', 'ozone', 'so2', 'nox', 'co', 'benzene']
urban_cities = ['London', 'Manchester', 'Birmingham', 'Leeds', 'Glasgow']
years = [2020, 2021, 2022, 2023, 2024, 2025]

# Step 4: Process each CSV
for file in files:
    file_path = os.path.join(folder_path, file)
    print(f"Processing {file}...")

    # Read CSV in chunks to handle large files (>1 GB)
    chunks = pd.read_csv(file_path, chunksize=100000, low_memory=False)
    for chunk in chunks:
        # Convert column names to lowercase for consistency
        chunk.columns = chunk.columns.str.lower()

        # Step 5: Separate by pollutant
        for pollutant in pollutants:
            if any(pollutant in col for col in chunk.columns):
                pol_col = next(col for col in chunk.columns if pollutant in col)
                cols = [col for col in chunk.columns if any(k in col for k in ['grid_id', 'station_id', 'lat', 'lon', 'local_authority', 'city', 'country', 'year', 'month', 'day', 'hour', 'date', 'status', 'quality'])]
                cols.append(pol_col)
                output_file = os.path.join(folder_path, f"{pollutant}_{file}")
                df_pol.to_csv(output_file, mode='a', index=False)

        # Step 6: Separate by year
        if any('year' in col for col in chunk.columns):
            year_col = next(col for col in chunk.columns if 'year' in col)
            for year in years:
                df_year = chunk[chunk[year_col] == year]
                if not df_year.empty:
                    output_file = os.path.join(folder_path, f"year_{year}_{file}")
                    df_year.to_csv(output_file, mode='a', index=False)

        # Step 7: Separate by urban local authority
        if any('local_authority' in col or 'city' in col for col in chunk.columns):
            auth_col = next(col for col in chunk.columns if 'local_authority' in col or 'city' in col)
            for city in urban_cities:
                df_city = chunk[chunk[auth_col].str.contains(city, case=False, na=False)]
                if not df_city.empty:
                    output_file = os.path.join(folder_path, f"{city}_{file}")
                    df_city.to_csv(output_file, mode='a', index=False)

        # Step 8: Derive health impacts (example: PM2.5 mortality)
        if any('pm2.5' in col for col in chunk.columns):
            pm25_col = next(col for col in chunk.columns if 'pm2.5' in col)
            chunk['estimated_deaths'] = chunk[pm25_col] * 0.006 * 1000  # Simplified COMEAP formula
            output_file = os.path.join(folder_path, f"health_{file}")
            chunk.to_csv(output_file, mode='a', index=False)

print("Cleaning complete! Check C:/Users/aloor/Downloads/CUrbanAirQuality for new CSV files.")