import duckdb

con = duckdb.connect('air_quality.duckdb')
sites = con.execute('SELECT DISTINCT "Site Name" FROM air_quality WHERE "Site Name" IS NOT NULL LIMIT 20').fetchdf()
print(sites) 