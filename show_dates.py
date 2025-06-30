import duckdb

print('Script started')
con = duckdb.connect('air_quality.duckdb')
try:
    print('Table columns:')
    cols = con.execute('PRAGMA table_info(air_quality)').fetchdf()
    print(cols)
except Exception as e:
    print('Error getting columns:', e)

try:
    print('Earliest dates:')
    earliest = con.execute("""
        SELECT DISTINCT "Date" FROM air_quality
        ORDER BY STRPTIME("Date", '%d-%m-%Y') ASC LIMIT 5
    """).fetchdf()
    print(earliest)
except Exception as e:
    print('Error getting earliest dates:', e)

try:
    print('Latest dates:')
    latest = con.execute("""
        SELECT DISTINCT "Date" FROM air_quality
        ORDER BY STRPTIME("Date", '%d-%m-%Y') DESC LIMIT 5
    """).fetchdf()
    print(latest)
except Exception as e:
    print('Error getting latest dates:', e) 