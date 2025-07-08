import fastf1
import pandas as pd
import os

# Enable F1 caching
cache_path = os.path.join(os.path.dirname(__file__), '../data/raw')
fastf1.Cache.enable_cache(cache_path)

# Load the session data
session = fastf1.get_session(2024, 'Silverstone', 'R')
session.load()

# Extract all laps with valid sector times
laps = session.laps
clean_laps = laps.loc[laps['LapTime'].notnull() & (laps['PitInTime'].isnull())]

# Keep only the relevant columns
data = clean_laps[['Driver', 'Team', 'LapNumber', 'LapTime', 'Stint', 
       'Compound', 'Sector1Time', 'Sector2Time', 'Sector3Time']]

# Convert time columns to seconds
time_cols = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']
for col in time_cols:
    data[col] = data[col].dt.total_seconds()

# Save the data to a CSV file
output_path = os.path.join(os.path.dirname(__file__), '../data/sector_data.csv')
data.to_csv(output_path, index=False)

print(f"✅ Extracted and saved {len(data)} laps to {output_path}")