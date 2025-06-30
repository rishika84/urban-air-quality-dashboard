import pandas as pd

# Read the CSV file (all columns, let pandas infer dtypes)
df = pd.read_csv('combined_cleaned_air_quality.csv')

# Save as Parquet for fast loading
df.to_parquet('combined_cleaned_air_quality.parquet', index=False)

print('Conversion to Parquet complete.') 