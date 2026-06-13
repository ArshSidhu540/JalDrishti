import pandas as pd

# 🚨 CHANGE THESE PATHS FOR YOUR NEW DOWNLOADED FILE
input_file = "C:/Users/arshd/Downloads/newly_downloaded_kaggle_file.csv"
output_file = "data/raw/ready_for_dashboard.csv"

# Read the file
df = pd.read_csv(input_file)

# --- EDIT THESE TO MATCH THE NEW CSV HEADERS ---
column_mapping = {
    'Station_Code': 'station_id',      # Change 'Station_Code' to match the file's column name
    'Lat': 'latitude',                 # Change 'Lat' to match the file's column name
    'Lon': 'longitude',                # Change 'Lon' to match the file's column name
    'Water_Level_Delta': 'target',     # Change 'Water_Level_Delta' to match the file's column name
    'Rainfall_mm': 'rainfall'          # Change 'Rainfall_mm' to match the file's column name
}

# Run the mapping conversion
df = df.rename(columns=column_mapping)

# Save it to your workspace
df.to_csv(output_file, index=False)
print(f"🎉 Success! Prepared file saved to {output_file}. You can now upload this directly via the browser!")