import pandas as pd

df = pd.read_csv("data/sales.csv")

print("First 5 rows:")
print(df.head())

print("\nSummary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

