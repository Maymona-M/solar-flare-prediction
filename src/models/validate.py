"""
validate.py  —  Asma Riyaz (60305750)
--------------------------------------
Loads both trained models, evaluates on test set, produces comparison
plots and a JSON summary. Matches column names in features_labeled.csv.

Usage:
    python src/models/validate.py \
        --baseline_dir outputs/baseline/ \
        --cnn_dir      outputs/cnn/ \
        --features     outputs/features_labeled.csv \
        --data_dir     data/processed/clean/ \
        --output       outputs/validation/
"""

import argparse, json, os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--baseline_dir", default="outputs/baseline/")
parser.add_argument("--cnn_dir",      default="outputs/cnn/")
parser.add_argument("--features",     default="outputs/features_labeled.csv")
parser.add_argument("--data_dir",     default="data/processed/clean/")
parser.add_argument("--output",       default="outputs/validation/")
parser.add_argument("--random_seed",  type=int, default=42)
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)
SEED     = args.random_seed
IMG_SIZE = (224, 224)
BATCH    = 32
FLARE_CLS = {"X", "M", "C"}

# ═══════════════════════════════════════════════════════════════════════════════
# 1. BASELINE — features_labeled.csv
# ═══════════════════════════════════════════════════════════════════════════════
print("[1/5] Loading baseline model & feature test set")
df = pd.read_csv(args.features)

# Binary label from 'target' or 'flare_class'
if "target" in df.columns:
    y_all = df["target"].values.astype(int)
else:
    y_all = df["flare_class"].apply(
        lambda x: 1 if str(x).upper() in FLARE_CLS else 0).values

feature_cols = [c for c in df.columns if c.startswith("feature_")]
X_all = df[feature_cols].fillna(0).values

_, X_feat_test, _, y_feat_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all)

scaler     = joblib.load(os.path.join(args.baseline_dir, "baseline_scaler.pkl"))
base_model = joblib.load(os.path.join(args.baseline_dir, "baseline_model.pkl"))
X_test_s   = scaler.transform(X_feat_test)
base_pred  = base_model.predict(X_test_s)
base_proba = base_model.predict_proba(X_test_s)[:, 1]
print(f"    Baseline test set: {len(y_feat_test)} samples")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CNN — images
# ═══════════════════════════════════════════════════════════════════════════════
print("[2/5] Loading CNN model & image test set")

def get_label(path):
    return 1 if os.path.basename(path)[0].upper() in FLARE_CLS else 0

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return tf.cast(img, tf.float32) / 255.0

all_paths  = [os.path.join(r, f)
              for r, _, files in os.walk(args.data_dir)
              for f in files if f.lower().endswith(".png")]
all_labels = np.array([get_label(p) for p in all_paths])

_, test_paths, _, y_cnn_test = train_test_split(
    all_paths, all_labels, test_size=0.2, random_state=SEED, stratify=all_labels)

test_ds = (tf.data.Dataset.from_tensor_slices((test_paths, y_cnn_test))
           .map(lambda p, l: (load_img(p), l), num_parallel_calls=tf.data.AUTOTUNE)
           .batch(BATCH).prefetch(tf.data.AUTOTUNE))

cnn_model = tf.keras.models.load_model(os.path.join(args.cnn_dir, "cnn_best.keras"))
cnn_proba = cnn_model.predict(test_ds).flatten()
cnn_pred  = (cnn_proba >= 0.5).astype(int)
print(f"    CNN test set: {len(y_cnn_test)} samples")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. METRICS
# ═══════════════════════════════════════════════════════════════════════════════
print("[3/5] Computing metrics")

def get_metrics(y_true, y_pred, y_proba, name):
    return {
        "model":    name,
        "f1":       round(f1_score(y_true, y_pred, average="weighted"), 4),
        "roc_auc":  round(roc_auc_score(y_true, y_proba), 4),
        "avg_prec": round(average_precision_score(y_true, y_proba), 4),
        "cm":       confusion_matrix(y_true, y_pred).tolist(),
        "report":   classification_report(y_true, y_pred,
                        target_names=["no-flare", "flare"]),
    }

base_m = get_metrics(y_feat_test, base_pred, base_proba, "Baseline (LR on features)")
cnn_m  = get_metrics(y_cnn_test,  cnn_pred,  cnn_proba,  "Full CNN (MobileNetV2)")

for m in [base_m, cnn_m]:
    print(f"\n{'='*55}\n{m['model']}\n{'='*55}")
    print(m["report"])
    print(f"F1 (weighted): {m['f1']}  |  ROC-AUC: {m['roc_auc']}  |  Avg Precision: {m['avg_prec']}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/5] Generating validation plots")
plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

COLORS = {"base": "#888780", "cnn": "#7F77DD"}

# ROC curves
ax1 = fig.add_subplot(gs[0, 0:2])
for (yt, yp, label, color) in [
    (y_feat_test, base_proba, "Baseline LR",       COLORS["base"]),
    (y_cnn_test,  cnn_proba,  "CNN (MobileNetV2)", COLORS["cnn"]),
]:
    fpr, tpr, _ = roc_curve(yt, yp)
    ax1.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(yt,yp):.3f})",
             color=color, lw=2)
ax1.plot([0,1],[0,1],"--", color="#B4B2A9", lw=1)
ax1.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
        title="ROC Curve — Baseline vs CNN", xlim=[0,1], ylim=[0,1.02])
ax1.legend()

# Metric bar chart
ax2 = fig.add_subplot(gs[0, 2])
mnames  = ["F1 (weighted)", "ROC-AUC", "Avg Precision"]
bvals   = [base_m["f1"], base_m["roc_auc"], base_m["avg_prec"]]
cvals   = [cnn_m["f1"],  cnn_m["roc_auc"],  cnn_m["avg_prec"]]
x = np.arange(len(mnames))
ax2.bar(x-0.2, bvals, 0.35, label="Baseline LR", color=COLORS["base"])
ax2.bar(x+0.2, cvals, 0.35, label="CNN",         color=COLORS["cnn"])
for i,(b,c) in enumerate(zip(bvals,cvals)):
    ax2.text(i-0.2, b+0.01, f"{b:.3f}", ha="center", fontsize=8)
    ax2.text(i+0.2, c+0.01, f"{c:.3f}", ha="center", fontsize=8)
ax2.set(xticks=x, xticklabels=mnames, ylim=[0,1.05],
        title="Metric Comparison", ylabel="Score")
ax2.legend(fontsize=9)

# Confusion matrices
for col, (cm_data, title) in enumerate([
    (base_m["cm"], "Baseline LR"),
    (cnn_m["cm"],  "CNN (MobileNetV2)"),
]):
    ax = fig.add_subplot(gs[1, col])
    cm = np.array(cm_data)
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=[0,1], yticks=[0,1],
           xticklabels=["no-flare","flare"],
           yticklabels=["no-flare","flare"],
           xlabel="Predicted", ylabel="Actual",
           title=f"Confusion Matrix — {title}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black",
                    fontsize=13, fontweight="bold")

# Precision-Recall
ax4 = fig.add_subplot(gs[1, 2])
for (yt, yp, label, color) in [
    (y_feat_test, base_proba, "Baseline LR",       COLORS["base"]),
    (y_cnn_test,  cnn_proba,  "CNN (MobileNetV2)", COLORS["cnn"]),
]:
    prec, rec, _ = precision_recall_curve(yt, yp)
    ax4.plot(rec, prec, color=color, lw=2,
             label=f"{label} (AP={average_precision_score(yt,yp):.3f})")
ax4.set(xlabel="Recall", ylabel="Precision",
        title="Precision-Recall Curve", xlim=[0,1], ylim=[0,1.02])
ax4.legend(fontsize=9)

plt.suptitle("Solar Flare Prediction — Phase 2 Validation Report",
             fontsize=14, fontweight="bold", y=1.01)

plot_path = os.path.join(args.output, "validation_report.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: {plot_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. SUMMARY JSON
# ═══════════════════════════════════════════════════════════════════════════════
print("[5/5] Saving summary")
summary = {
    "baseline": {k:v for k,v in base_m.items() if k != "report"},
    "cnn":      {k:v for k,v in cnn_m.items()  if k != "report"},
    "hypothesis_supported": cnn_m["roc_auc"] > base_m["roc_auc"],
    "improvement_roc_auc":  round(cnn_m["roc_auc"] - base_m["roc_auc"], 4),
}
with open(os.path.join(args.output, "validation_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nValidation complete. Outputs: {args.output}")
print(f"Hypothesis supported: {summary['hypothesis_supported']}")
print(f"ROC-AUC improvement:  {summary['improvement_roc_auc']:+.4f}")