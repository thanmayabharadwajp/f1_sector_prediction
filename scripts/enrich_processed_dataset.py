import pandas as pd
import os

# === Load both datasets ===
features_path = os.path.join(os.path.dirname(__file__), '../data/sector_data.csv')
processed_path = os.path.join(os.path.dirname(__file__), '../data/processed_dataset.csv')
enriched_path = os.path.join(os.path.dirname(__file__), '../data/processed_dataset_enriched.csv')

raw_df = pd.read_csv(features_path)
processed_df = pd.read_csv(processed_path)

# === Match length first ===
if len(processed_df) > len(raw_df):
    raise ValueError("Processed dataset is longer than raw dataset. Can't align safely.")

# Drop rows from raw_df that were removed during lag feature generation
raw_df = raw_df.iloc[-len(processed_df):].reset_index(drop=True)
processed_df = processed_df.reset_index(drop=True)

# === Merge columns ===
processed_df['DriverName'] = raw_df['Driver']
processed_df['LapNumber'] = raw_df['LapNumber']

# === Save enriched file ===
processed_df.to_csv(enriched_path, index=False)
print(f"✅ Enriched dataset saved to {enriched_path} with shape {processed_df.shape}")
