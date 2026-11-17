from pathlib import Path
import pandas as pd

project_folder = Path(__file__).resolve().parent
dataset_path = project_folder / "data" / "real_estate.xlsx"

df = pd.read_excel(dataset_path)

print("\n--- DATASET SIZE ---")
print(df.shape)

print("\n--- FIRST 5 ROWS ---")
print(df.head().to_string(index=False))

print("\n--- COLUMN NAMES ---")
for column in df.columns:
    print(column)

print("\n--- MISSING VALUES ---")
print(df.isna().sum())