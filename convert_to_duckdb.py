import duckdb
import pandas as pd
import re

# Read the CSV file
csv_file = 'combined_cleaned_air_quality.csv'
df = pd.read_csv(csv_file)

# Only keep rows with valid Date and Site Name
pattern = re.compile(r'^[0-9]{2}-[0-9]{2}-[0-9]{4}$')
df = df[df['Date'].astype(str).apply(lambda x: bool(pattern.match(x)))]
df = df[pd.notna(df['Site Name'])]

# Create DuckDB database and write the table
con = duckdb.connect('air_quality.duckdb')
con.execute('DROP TABLE IF EXISTS air_quality')
con.execute('CREATE TABLE air_quality AS SELECT * FROM df')
con.close()

print('DuckDB database created as air_quality.duckdb with only valid rows.') 