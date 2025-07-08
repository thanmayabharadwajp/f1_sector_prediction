# === Team color mapping (F1 2024 team colors)
TEAM_COLORS = {
    "Red Bull Racing": "#1E41FF",
    "Ferrari": "#DC0000",
    "Mercedes": "#00D2BE",
    "McLaren": "#FF8700",
    "Aston Martin": "#006F62",
    "Alpine": "#0090FF",
    "AlphaTauri": "#2B4562",
    "Williams": "#005AFF",
    "Alfa Romeo": "#900000",
    "Haas F1 Team": "#B6BABD"
}

def get_team_color(team_name):
    return TEAM_COLORS.get(team_name, "#CCCCCC")  # fallback to grey

import streamlit as st
import pandas as pd
import joblib
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# === Configuration ===
pause_time = 1.2
FEATURES = [
    'Driver', 'Team', 'Compound', 'Stint',
    'PrevSector1', 'PrevSector2', 'PrevSector3', 'PrevLapTime'
]
TARGET = 'Sector2Time'

# === Load model and data ===
@st.cache_resource
def load_model():
    return joblib.load("models/sector_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed_dataset_enriched.csv")

model = load_model()
df = load_data()

# === Sidebar: Select driver ===
drivers = df['DriverName'].unique()
selected_driver = st.sidebar.selectbox("Select Driver", sorted(drivers))
mode = st.sidebar.radio("Mode", ["Play (Auto)", "Manual"])

# === Filter and sort
driver_df = df[df['DriverName'] == selected_driver].sort_values(by='LapNumber').reset_index(drop=True)
total_laps = len(driver_df)

# === Session State
if "lap_index" not in st.session_state:
    st.session_state.lap_index = 0
if "errors" not in st.session_state:
    st.session_state.errors = []

# === Layout
st.title("🏎️ F1 Sector 2 Prediction Dashboard")
st.markdown(f"**Driver:** `{selected_driver}` | Laps: {total_laps}")

# === Reset Button
if st.button("🔄 Reset"):
    st.session_state.lap_index = 0
    st.session_state.errors = []

# === Simulate lap-by-lap
if st.session_state.lap_index < total_laps:
    row = driver_df.iloc[st.session_state.lap_index]
    X_input = pd.DataFrame([row[FEATURES].values], columns=FEATURES)
    predicted = model.predict(X_input)[0]
    actual = row[TARGET]
    delta = predicted - actual

    st.session_state.errors.append(delta)

    # === Metrics Row
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Sector 2", f"{predicted:.3f}s")
    col2.metric("Actual Sector 2", f"{actual:.3f}s")
    col3.metric("Δ (Error)", f"{delta:+.3f}s", delta_color="inverse")

    # === Lap info
    st.caption(f"Lap {int(row['LapNumber'])} — Compound: `{row['Compound']}` — Stint: `{row['Stint']}`")

    # === Progress bar
    st.progress((st.session_state.lap_index + 1) / total_laps)

    # === Error Line Chart
    with st.expander("📈 Live Prediction Error (Δ)"):
        fig, ax = plt.subplots()
        ax.plot(st.session_state.errors, marker='o', linestyle='-', label='Δ Error')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel("Lap Index")
        ax.set_ylabel("Δ Time (s)")
        ax.set_title("Prediction Error Over Time")
        ax.grid(True)
        st.pyplot(fig)

        # === Telemetry-style Sector Trace Plot
    with st.expander("📡 Telemetry: Sector Time Trends"):
        sector_data = driver_df.iloc[:st.session_state.lap_index+1]
        laps = sector_data["LapNumber"]
        fig2, ax2 = plt.subplots()

        ax2.plot(laps, sector_data["PrevSector1"], label="Sector 1", linestyle='--')
        ax2.plot(laps, sector_data["PrevSector2"], label="Sector 2 (Previous)", linestyle='--')
        ax2.plot(laps, sector_data["PrevSector3"], label="Sector 3", linestyle='--')
        ax2.plot(laps, sector_data["Sector2Time"], label="Actual Sector 2", color="black", linewidth=2)
        predicted_vals = model.predict(sector_data[FEATURES])
        ax2.plot(laps, predicted_vals, label="Predicted Sector 2", color=get_team_color(row['Team']), linewidth=2)

        ax2.set_xlabel("Lap Number")
        ax2.set_ylabel("Sector Time (s)")
        ax2.set_title("Sector Time Telemetry")
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

    # === Advance lap
    if mode == "Play (Auto)":
        st.session_state.lap_index += 1
        time.sleep(pause_time)
        st.rerun()
    elif st.button("Next Lap ▶️"):
        st.session_state.lap_index += 1
else:
    st.success("✅ All laps simulated!")

    # === Final stats
    st.subheader("📊 Driver Summary")
    mae = mean_absolute_error(driver_df[TARGET], model.predict(driver_df[FEATURES]))
    from math import sqrt
    rmse = sqrt(mean_squared_error(driver_df[TARGET], model.predict(driver_df[FEATURES])))
    st.metric("MAE", f"{mae:.3f}s")
    st.metric("RMSE", f"{rmse:.3f}s")
