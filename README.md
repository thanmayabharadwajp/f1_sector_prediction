# 🏁 F1 Sector Time Prediction Dashboard

This project uses machine learning to predict Sector 2 lap times in Formula 1 races based on historical FastF1 telemetry data. Built with Streamlit for real-time visualization.
---

## 🚦 Features

- Predicts Sector 2 time per lap for any driver
- Real-time Streamlit dashboard (Play / Manual mode)
- Visual telemetry traces (Sector trends)
- Team-colored prediction overlays
- MAE & RMSE evaluation summary

---

## 📊 Tech Stack

- `FastF1` – F1 timing and telemetry data
- `scikit-learn` – RandomForestRegressor model
- `Streamlit` – Interactive real-time dashboard
- `pandas`, `matplotlib`, `joblib`

---

## 🧠 ML Features

The model uses:
- Driver, Team, Tyre compound
- Stint number
- Previous sector times and lap time

To predict:
- Sector2Time

---
