import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

model_dir = "outputs/baseline"
data_path = "outputs/features_labeled.csv"
output_path = "outputs/predictions.csv"

print(f"[{datetime.now()}] Loading model...")
model = joblib.load(os.path.join(model_dir, "model.pkl"))
scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
print(f"[{datetime.now()}] Model loaded successfully")

df = pd.read_csv(data_path)
feature_cols = [f"feature_{i}" for i in range(30)]
X = df[feature_cols].fillna(0).values
X_scaled = scaler.transform(X)

print(f"[{datetime.now()}] Running inference on {len(df)} rows...")
preds = model.predict(X_scaled)
proba = model.predict_proba(X_scaled)[:, 1]

results = pd.DataFrame({
    "filename": df["filename_x"],
    "prediction": preds,
    "probability": proba,
    "timestamp": datetime.now().isoformat()
})
results.to_csv(output_path, index=False)
print(f"[{datetime.now()}] Done. Predictions saved to {output_path}")
print(f"Flare predictions: {preds.sum()} positives out of {len(preds)}")
