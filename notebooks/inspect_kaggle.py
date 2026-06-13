import pandas as pd

# Path to the real dataset you just added
real_csv_path = "data/raw/india_groundwater_2025.csv"  # Check your exact filename!

try:
    df = pd.read_csv(real_csv_path, nrows=5)
    print("🎯 KAGGLE DATASET DETECTED SUCCESSFULLY!")
    print("\n--- Available Columns in the Dataset ---")
    print(df.columns.tolist())
    print("\n--- First Row Preview ---")
    print(df.head(1))
except Exception as e:
    print(f"❌ Error reading file: {e}. Please make sure your filename in data/raw matches perfectly.")