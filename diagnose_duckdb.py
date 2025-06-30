import duckdb

con = duckdb.connect('air_quality.duckdb')

total_rows = con.execute('SELECT COUNT(*) FROM air_quality').fetchone()[0]
valid_rows = con.execute('''
    SELECT COUNT(*) FROM air_quality
    WHERE REGEXP_MATCHES("Date", '^[0-9]{2}-[0-9]{2}-[0-9]{4}$')
      AND "Site Name" IS NOT NULL
''').fetchone()[0]
sample = con.execute('SELECT * FROM air_quality LIMIT 5').fetchdf()

print(f'Total rows: {total_rows}')
print(f'Valid rows: {valid_rows}')
print('Sample rows:')
print(sample) 