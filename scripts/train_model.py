import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random 
import joblib
import matplotlib.pyplot as plt
from math import sqrt

# Load data
df_path = os.path.join(os.path.dirname(__file__), '../data/processed_dataset.csv')
df = pd.read_csv(df_path)

# Load target and features
features = ['Driver', 'Team', 'Compound', 'Stint', 'PrevSector1', 'PrevSector2', 'PrevSector3', 'PrevLapTime']
target = 'Sector2Time'

X = df[features]
y = df[target]

# Split data into the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred =model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))

print(f"Model trained with MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# Save model
model_path = os.path.join(os.path.dirname(__file__), '../models/sector_model.pkl')
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# Plot the predictions vs actual values
plt.figure(figsize=(8,8))
plt.scatter(y_test, y_pred, alpha = 0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Sector 2 Time')
plt.ylabel('Predicted Sector 2 Time')
plt.title('Actual vs Predicted Sector 2 Time')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), '../models/scatter_plot.png'))
plt.show()


