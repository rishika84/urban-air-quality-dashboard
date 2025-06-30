import pandas as pd
import sqlite3
import os

# File paths
parquet_file = 'combined_cleaned_air_quality.parquet'
sqlite_file = 'air_quality.sqlite'
table_name = 'air_quality'

# Load data
print('Loading Parquet file...')
df = pd.read_parquet(parquet_file)
df = df.rename(columns=lambda x: x.strip())
df = df[df['Date'].astype(str).str.match(r'^[0-9]{2}-[0-9]{2}-[0-9]{4}$')]
df = df[pd.notna(df['Site Name'])]
df['Date'] = df['Date'].astype(str)
df['Time'] = df['Time'].astype(str)
df['Site Name'] = df['Site Name'].astype(str).str.strip()
def fix_time(row):
    if row['Time'] == '24:00:00':
        date_obj = pd.to_datetime(row['Date'], format='%d-%m-%Y', errors='coerce')
        if pd.notnull(date_obj):
            date_obj += pd.Timedelta(days=1)
            return pd.Timestamp(f"{date_obj.strftime('%Y-%m-%d')} 00:00:00")
    return pd.to_datetime(row['Date'] + ' ' + row['Time'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
df['Datetime'] = df.apply(fix_time, axis=1)
df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
df = df.dropna(subset=['Datetime'])
pollutant_cols = [col for col in ['Nitrogen dioxide', 'PM10', 'PM2.5'] if col in df.columns]
for col in pollutant_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=pollutant_cols, how='all')
df = df.sort_values('Datetime').reset_index(drop=True)

# Select columns for SQLite
cols = ['Site Name', 'Datetime', 'PM2.5', 'PM10', 'Nitrogen dioxide']
df = df[cols]
df = df.rename(columns={'Site Name': 'site', 'Datetime': 'datetime', 'PM2.5': 'pm25', 'PM10': 'pm10', 'Nitrogen dioxide': 'no2'})

# Write to SQLite
def to_sqlite(df, db_file, table_name):
    if os.path.exists(db_file):
        os.remove(db_file)
    print(f'Writing to SQLite database: {db_file}')
    conn = sqlite3.connect(db_file)
    df.to_sql(table_name, conn, index=False)
    # Create indexes
    print('Creating indexes...')
    conn.execute(f'CREATE INDEX idx_site ON {table_name} (site);')
    conn.execute(f'CREATE INDEX idx_datetime ON {table_name} (datetime);')
    conn.commit()
    conn.close()
    print('Done.')

to_sqlite(df, sqlite_file, table_name) 