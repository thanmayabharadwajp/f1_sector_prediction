import pandas as pd
import joblib
import os
import time

# === Config ===
pause_time = 1.5
FEATURES = [
    'Driver', 'Team', 'Compound', 'Stint',
    'PrevSector1', 'PrevSector2', 'PrevSector3', 'PrevLapTime'
]
TARGET = 'Sector2Time'

# === Load Model and Data ===
df = pd.read_csv("data/processed_dataset_enriched.csv")
model = joblib.load("models/sector_model.pkl")

# === Simulation Loop ===
print("\nStarting Sector 2 Prediction Simulation...\n")
for i, row in df.iterrows():
    X_input = pd.DataFrame([row[FEATURES]], columns=FEATURES)
    actual = row[TARGET]
    predicted = model.predict(X_input)[0]
    delta = predicted - actual

    print(f"Lap {int(row['LapNumber'])} | Driver {row['DriverName']} | Predicted: {predicted:.3f}s | Actual: {actual:.3f}s | Δ: {delta:+.3f}s")
    time.sleep(pause_time)

print("\n✅ Simulation complete.")
