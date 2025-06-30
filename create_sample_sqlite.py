import sqlite3
import pandas as pd
import os

SOURCE_DB = 'air_quality.sqlite'
SAMPLE_DB = 'air_quality_sample.sqlite'
TABLE = 'air_quality'
SAMPLE_SIZE = 2000  # You can change this number

# Connect to source DB and sample data
with sqlite3.connect(SOURCE_DB) as conn:
    df = pd.read_sql(f'SELECT * FROM {TABLE}', conn)

# Take a random sample
sample_df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)

# Write to new SQLite DB
def to_sqlite(df, db_file, table_name):
    if os.path.exists(db_file):
        os.remove(db_file)
    with sqlite3.connect(db_file) as conn:
        df.to_sql(table_name, conn, index=False)
        conn.execute(f'CREATE INDEX idx_site ON {table_name} (site);')
        conn.execute(f'CREATE INDEX idx_datetime ON {table_name} (datetime);')
        conn.commit()
    print(f'Sample database created: {db_file} ({len(df)} rows)')

to_sqlite(sample_df, SAMPLE_DB, TABLE) 