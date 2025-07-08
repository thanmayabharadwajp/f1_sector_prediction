import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# Load day 1 data
df_path = os.path.join(os.path.dirname(__file__), '../data/sector_data.csv')
df = pd.read_csv(df_path)

# Sort by Driver and LapNumber to calculate lag features
df = df.sort_values(by=['Driver', 'LapNumber'])

# Add lag features (from previous lap)
df['PrevSector1'] = df.groupby('Driver')['Sector1Time'].shift(1)
df['PrevSector2'] = df.groupby('Driver')['Sector2Time'].shift(1)
df['PrevSector3'] = df.groupby('Driver')['Sector3Time'].shift(1)
df['PrevLapTime'] = df.groupby('Driver')['LapTime'].shift(1)

# Drop rows with NaN values in lag features
df = df.dropna(subset=['PrevSector1', 'PrevSector2', 'PrevSector3', 'PrevLapTime'])

# Encode categorical features
label_encoders = {}
for col in ['Driver', 'Team', 'Compound']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Select features and target
features = ['Driver', 'Team', 'Compound', 'Stint', 'PrevSector1', 'PrevSector2', 'PrevSector3', 'PrevLapTime']
target = 'Sector2Time'

X = df[features]
y = df[target]

# Save full processed dataset for training
processed_path = os.path.join(os.path.dirname(__file__), '../data/processed_dataset.csv')
df[features + [target]].to_csv(processed_path, index=False)

print(f"✅ Processed dataset saved with shape {df.shape} to {processed_path}")