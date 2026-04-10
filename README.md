# Solar Flare Prediction — Phase 2

**Course:** DSAI 3202 — Data Pipeline, ETL, and Feature Foundations  
**Institution:** University of Doha for Science and Technology

---

## Team

| Name | Student ID | Role |
|---|---|---|
| Maymona Mustafa | 60306027 | Data Ingestion, Data Cataloging |
| Asma Riyaz | 60305750 | ETL Pipeline, Exploratory Analysis, Feature Extraction |

---

## Project Description

This project develops a supervised machine learning system to predict solar flare occurrence using magnetogram images of solar active regions. The system ingests large-scale image data from a publicly accessible scientific repository, preprocesses and normalizes the images, and extracts features for use in downstream classification models.

**Hypothesis:** A supervised model trained on full-resolution solar magnetogram images will predict solar flare occurrence more accurately than a baseline model trained on reduced or simplified image features.

**Dataset:** Active Region Magnetograms for Solar Flare Prediction (Zenodo, NASA SDO/HMI)  
**Source:** https://doi.org/10.5281/zenodo.7775776  
**Size:** ~13 GB · ~950,000 PNG images · 224×224 resolution

---

## Phase 2: Model Development, Validation & Deployment

### Phase 2 Objectives

Phase 2 focuses on model development, validation, and deployment within an AI system. The objectives are:

1. **Model Development** — Select appropriate model(s) aligned with the project hypothesis. Establish baseline and full-feature models with reproducible training settings including data splits, hyperparameters, and random seeds.
2. **Model Validation** — Define and apply a validation strategy appropriate to the problem context. Select evaluation metrics, analyze errors, and compare against baselines while avoiding data leakage.
3. **Model Versioning and Registration** — Version trained models and register them in Azure ML with metadata including training data version, feature set, metrics, and limitations.
4. **Deployment** — Deploy the selected model using batch serving mode with clear input/output interfaces and documented deployment configuration.
5. **Deployment Validation** — Verify deployed model behavior through functional tests, confirming consistency between offline training and deployed predictions.


### Azure Infrastructure (Phase 2)

| Resource | Name | Description |
|----------|------|-------------|
| Resource Group | `rg-60306027` | Container for all Azure resources |
| Storage Account | `solarflarestorageproject` | Raw, processed, and curated image storage |
| AML Workspace | `solar-flare-aml-60306027` | Central ML resource management |
| Compute Cluster | `solar-flare-cluster` | Standard_DS3_v2, 0-2 nodes, auto-scale |
| Environment | `solar-flare-env:2` | PyTorch 2.0.1, TensorFlow, scikit-learn, pandas, azureml-core |
| Labeled Dataset | `solar-flare-labeled:1` | Versioned dataset with 950,047 labeled samples |
| CNN Model | `solar-flare-cnn:1` | MobileNetV2-based model (26 MB) |
| Batch Endpoint | `solar-flare-batch-endpoint` | Real-time inference endpoint |


---

## Repository Structure
solar-flare-prediction/
├── src/                    # Training scripts, CNN model
├── src_v2/                 # Inference scripts
│   └── batch_inference.py  # Main inference entry point
├── azure/                  # Azure ML configuration
│   ├── batch_endpoint.yml
│   ├── batch_deployment.yml
│   ├── inference_job.yml
│   └── aml_environment.yml
├── .azure-pipelines/
│   └── solar-flare-pipeline.yml  # DevOps CI pipeline
├── outputs/
│   ├── baseline/           # Trained model artifacts (model.pkl, scaler.pkl)
│   └── predictions.csv     # Latest inference output
└── README.md

---


### Phase 2 Implementation

#### Labeled Dataset Preparation

The Phase 1 features were merged with official Dryad labels to create a properly labeled dataset for model training.

| Source | Description |
|--------|-------------|
| Features | `Lat60_Lon60_Nans0_C1.0_24hr_png_224_features.csv` (950,047 rows) |
| Labels | `C1.0_24hr_224_png_Labels.txt` from Dryad (filename, flare_class) |
| Merge Strategy | Filename-based join using base filename |
| Output | `features_labeled.csv` (950,047 rows, 31 features + flare_class + binary target) |

**Class Distribution:**
- Flare positive (≥C1.0): 878,485 images (92.5%)
- Flare negative: 71,562 images (7.5%)

---

## Model Development

### Baseline Model
- **Algorithm:** Logistic Regression (scikit-learn)
- **Justification:** Chosen as an interpretable, fast-training baseline suitable for binary classification with tabular features. Provides a performance floor for comparison against more complex models.
- **Features:** 30 CNN-derived image features (`feature_0` to `feature_29`) extracted from solar magnetogram images
- **Target:** Binary — flare (1) vs no-flare (0)
- **Random seed:** Fixed for reproducibility
- **Data split:** Train/validation/test with stratification on target label

### CNN Model
- A Convolutional Neural Network was also developed for direct image-based classification
- Architecture defined in `src/cnn_model.py`
- Trained using PyTorch with fixed random seeds

---

## Model Validation

### Strategy
- Stratified train/validation/test split to preserve class balance
- Evaluation on held-out test set only — no data leakage
- Metrics chosen to reflect realistic class imbalance in solar flare data

**CNN Model Results (Test Set):**

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.9999** |
| Accuracy | **94%** |
| Precision (Flare) | 1.00 |
| Recall (Flare) | 0.93 |
| F1-Score (Flare) | 0.97 |

**Confusion Matrix:**

|                | Predicted No-Flare | Predicted Flare |
|----------------|--------------------|-----------------|
| Actual No-Flare | 65                 | 0               |
| Actual Flare   | 61                 | 874             |

**Key Findings:**
- **Perfect Precision**: No false positives (all 65 no-flare images correctly identified)
- **High Recall**: 93% of flares correctly identified (874/935)
- **Near-Perfect AUC**: 0.9999 indicates excellent class separation
- **Model Generalization**: Strong performance on unseen test data

---

## Model Versioning and Registration

All artifacts are registered in Azure ML:

| Asset | Name | Version |
|---|---|---|
| Model | `solar-flare-model` | 1 |
| Environment | `solar-flare-env` | 3 |
| Data (features) | `solar-flare-features-labeled` | 1 |
| Data (with target) | `solar-flare-features-target` | 1 |
| Data (with metadata) | `solar-flare-features-metadata` | 1 |

- **Workspace:** `Amazon-Electronics-Lab-60305750`
- **Resource Group:** `rg-60305750`
- **Subscription:** `0b475409-9d7c-4dff-a07b-084eff651874`

---

## Deployment

### Serving Mode
Batch inference using Azure ML CommandJob — appropriate for periodic solar flare prediction over datasets of observations.

### Endpoint
- **Batch Endpoint:** `solar-flare-endpoint-60305750`
- **Scoring URI:** `https://solar-flare-endpoint-60305750.qatarcentral.inference.ml.azure.com/jobs`
- **Deployment:** `solar-flare-deploy-60305750`
- **Compute:** `cpu-cluster` (Standard_DS3_v2)

### Input/Output Interface
- **Input:** CSV file with columns `feature_0` to `feature_29`
- **Output:** `predictions.csv` with columns: `filename`, `prediction` (0/1), `probability` (float), `timestamp`

### Feature Parity
Training and serving use identical feature columns (`feature_0` to `feature_29`) with the same `StandardScaler` fitted during training and saved as `scaler.pkl`.

### Inference Script
- `src_v2/batch_inference.py` — loads model and scaler from AML Model Registry, downloads input data from registered datastore, runs inference, saves predictions

---

## Deployment Validation

### Functional Test
- Inference job `dreamy_king_6f2985n2fr` ran successfully on `features_labeled.csv`
- Status: **Completed** ✅
- Output: `predictions.csv` generated with prediction and probability per row
- Verified via: `az ml job show --name dreamy_king_6f2985n2fr --query status`

### Sanity Checks
- Model loads without errors from registered artifact
- Scaler transforms 30 features correctly
- Predictions are binary (0/1) with probabilities in [0,1]
- No data leakage — scaler fitted on training data only

---

## DevOps Automation

### Pipeline
- **File:** `.azure-pipelines/solar-flare-pipeline.yml`
- **Trigger:** Push to `phase2` branch
- **Service Connection:** `SC-UDST-CCIT-DSAI3202-2`
- **Latest successful run:** `#20260409.17` ✅

### What it does
On every push to `phase2`, the pipeline:
1. Authenticates to Azure using the service principal
2. Installs the Azure ML CLI extension
3. Submits `azure/inference_job.yml` as a CommandJob to AML
4. Streams logs and reports success/failure

---

## Rollback Plan

To revert to a previous model version:
1. Register the previous model: `az ml model create --name solar-flare-model --version <N> --path outputs/baseline/`
2. Update `azure/batch_deployment.yml` to point to the previous version: `model: azureml:solar-flare-model:<N>`
3. Redeploy: `az ml batch-deployment create --file azure/batch_deployment.yml --set-default`
4. Verify with a test invocation before routing production traffic

---

## Azure Infrastructure

| Resource | Name |
|---|---|
| Resource Group | rg-60306027 |
| Storage Account | solarflarestorageproject |
| Raw Container | raw |
| Processed Container | processed |
| Curated Container | curated |
| AML Workspace | solar-flare-aml-60306027 |

---

## How to Run Inference Manually

```bash
az ml job create \
  --file azure/inference_job.yml \
  --resource-group rg-60305750 \
  --workspace-name Amazon-Electronics-Lab-60305750 \
  --stream
```

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Maymona-M/solar-flare-prediction.git
cd solar-flare-prediction
git checkout phase2
```

### 2. Create the environment
```bash
conda env create -f environment.yml
conda activate solar-flare-env
```

### 3. Set credentials
```bash
export AZURE_STORAGE_KEY="your_storage_account_key"
```

### 4. Run the pipeline
```bash
# Ingest data (Phase 1)
python src/ingestion/download_dataset.py

# Run ETL (Phase 1)
python src/etl/preprocess.py

# Train CNN model (Phase 2)
python src/train_cnn.py --features_path data/features_labeled.csv --epochs 10

# Or submit to Azure ML
az ml job create -f pipelines/cnn_training_pipeline.yml
```

---

## Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, submitted snapshots |
| `phase1` | Phase 1 development (data foundation) |
| `phase2` | Phase 2 development (modeling & deployment) |
---

## Citation

Boucheron, L. E., Vincent, T., Grajeda, J. A., & Wuest, E. (2023). Active Region Magnetograms for Solar Flare Prediction: Reduced Resolution Dataset Images. Zenodo. https://doi.org/10.5281/zenodo.7775776
