"""
train_baseline.py  —  Asma Riyaz (60305750)
--------------------------------------------
Baseline model: Logistic Regression on Phase 1 extracted features.
Reads features_labeled.csv, uses 'target' column (binary: 1=flare, 0=no-flare).

Usage:
    python src/models/train_baseline.py \
        --features outputs/features_labeled.csv \
        --output   outputs/baseline/
"""

import argparse, json, os, time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--features",    default="outputs/features_labeled.csv")
parser.add_argument("--output",      default="outputs/baseline/")
parser.add_argument("--test_size",   type=float, default=0.2)
parser.add_argument("--random_seed", type=int,   default=42)
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)
SEED = args.random_seed
np.random.seed(SEED)

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"[1/5] Loading features from {args.features}")
df = pd.read_csv(args.features)

print(f"    Columns found: {df.columns.tolist()}")

# Use 'target' as binary label (already 0/1 in your CSV)
# Fallback: derive from 'flare_class' if 'target' is missing
if "target" in df.columns:
    y = df["target"].values.astype(int)
elif "flare_class" in df.columns:
    y = df["flare_class"].apply(lambda x: 1 if str(x).upper() in ["X","M","C"] else 0).values
else:
    raise ValueError("No 'target' or 'flare_class' column found.")

# All feature_N columns
feature_cols = [c for c in df.columns if c.startswith("feature_")]
X = df[feature_cols].fillna(0).values

print(f"    Samples: {len(X)} | Features: {len(feature_cols)}")
print(f"    Flare (1): {y.sum()} | No-flare (0): {(y==0).sum()}")

# Show flare_class distribution if available
if "flare_class" in df.columns:
    print(f"    Flare class breakdown:\n{df['flare_class'].value_counts().to_string()}")

# ── Split ─────────────────────────────────────────────────────────────────────
print("\n[2/5] Splitting 80/20 (stratified)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=SEED, stratify=y)

# ── Scale ─────────────────────────────────────────────────────────────────────
print("[3/5] Scaling features")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Train ─────────────────────────────────────────────────────────────────────
print("[4/5] Training Logistic Regression")
t0 = time.time()
model = LogisticRegression(
    max_iter=1000, class_weight="balanced",
    random_state=SEED, solver="lbfgs", C=1.0)
model.fit(X_train_s, y_train)
print(f"    Done in {time.time()-t0:.2f}s")

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("[5/5] Evaluating")
y_pred  = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:, 1]
f1      = f1_score(y_test, y_pred, average="weighted")
roc_auc = roc_auc_score(y_test, y_proba)

print("\n" + "="*50)
print("BASELINE RESULTS")
print("="*50)
print(classification_report(y_test, y_pred, target_names=["no-flare","flare"]))
print(f"F1 (weighted): {f1:.4f}")
print(f"ROC-AUC:       {roc_auc:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(model,  os.path.join(args.output, "baseline_model.pkl"))
joblib.dump(scaler, os.path.join(args.output, "baseline_scaler.pkl"))

metadata = {
    "model_type":       "LogisticRegression",
    "feature_cols":     feature_cols,
    "n_features":       len(feature_cols),
    "n_train":          int(len(X_train)),
    "n_test":           int(len(X_test)),
    "random_seed":      SEED,
    "f1_weighted":      round(f1, 4),
    "roc_auc":          round(roc_auc, 4),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "class_mapping":    {"0": "no-flare", "1": "flare"},
}
with open(os.path.join(args.output, "baseline_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nArtifacts saved to: {args.output}")