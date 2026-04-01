# Solar Flare Prediction

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

## Phase 1 Objectives

Phase 1 focuses on building the data foundation of the AI system. The objectives are:

1. **Data Ingestion** — Identify data sources and implement a reliable batch ingestion process. Raw data is preserved and versioned in Azure Blob Storage.
2. **ETL Process** — Design and implement an automated ETL pipeline that cleans, validates, and normalizes the raw magnetogram images. All transformations are reproducible.
3. **Cataloging and Governance** — Register the dataset and its schema in a data catalog. Document schema definitions, data types, lineage, and storage zones.
4. **Exploratory Analysis** — Conduct exploratory data analysis to assess class distributions, pixel intensity patterns, outliers, and data readiness.
5. **Feature Extraction** — Define and implement an initial set of features aligned with the project hypothesis, including statistical, gradient, and spatial features.

---

## Repository Structure
```
solar-flare-prediction/
├── src/
│   ├── ingestion/         # Data ingestion scripts
│   ├── etl/               # ETL pipeline scripts
│   ├── features/          # Feature extraction scripts
│   └── catalog/           # Data catalog and schema
├── notebooks/
│   ├── EDA.ipynb          # Exploratory data analysis
│   └── ETL.ipynb          # ETL pipeline notebook
├── azure/
│   └── catalog.json       # Azure data catalog metadata
├── datastores/            # Azure datastore config (not committed)
├── environment.yml        # Conda environment
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Phase 1 Implementation

### Data Ingestion

- **Source:** Zenodo record 7775776 (NASA Solar Dynamics Observatory)
- **Mode:** Batch ingestion
- **Format:** PNG images (224×224 grayscale magnetograms)
- **Script:** `src/ingestion/download_dataset.py`
- **Storage:** Azure Blob Storage — `solarflarestorageproject` — `raw` container
- **Layout:** `raw/magnetograms/<label>_<timestamp>.png`

The ingestion script downloads the dataset using `zenodo_get`, extracts the archive, and uploads all PNG files to the raw container using the Azure Blob Storage SDK.

### ETL Process

- **Script:** `src/etl/preprocess.py`
- **Notebook:** `notebooks/ETL.ipynb`
- **Steps:**
  1. Read images from raw container
  2. Validate each image (size check, blank detection, NaN check)
  3. Normalize pixel values to 0–1 range
  4. Upload cleaned images to processed container (`clean/` prefix)
  5. Generate validation report CSV

Invalid images are logged and skipped. All transformations are reproducible and parameterized.

### Cataloging and Governance

- **Catalog:** `azure/catalog.json`
- **Schema:** Grayscale PNG, 224×224, uint8 pixel values
- **Zones:** `raw/` (original), `processed/clean/` (validated and normalized)
- **Labels:** X (strongest), M (moderate), C (weak), B (very weak), N (no flare)
- **Lineage:** Zenodo → raw container → ETL → processed container → feature store

### Exploratory Analysis

- **Notebook:** `notebooks/EDA.ipynb`
- **Covers:**
  - Class distribution (multi-class and binary)
  - Pixel intensity distributions by flare class
  - Mean and standard deviation analysis per class
  - Sample image visualization grid
  - Outlier detection via blank image flagging

### Feature Extraction

- **Script:** `src/features/feature_extraction.py`
- **Features extracted per image:**
  - **Statistical:** mean, std, min, max, skewness, kurtosis, IQR, percentiles
  - **Gradient:** mean/std/max of pixel gradient magnitude (edge strength)
  - **Spatial:** center of mass (x, y), active region pixel ratio
- **Output:** `outputs/features.csv`
- **Justification:** Magnetic field intensity statistics capture flare-relevant patterns; gradient features detect sharp magnetic polarity boundaries associated with flare activity; spatial features capture active region geometry.

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


## Repository Structure
```
solar-flare-prediction/
├── README.md                      # Complete project documentation
├── environment.yml                # Conda environment (Python 3.9)
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
│
├── src/                           # Source code
│   ├── train.py                   # Logistic Regression training script
│   ├── train_cnn.py               # CNN training script (MobileNetV2)
│   ├── score.py                   # Batch endpoint scoring script
│   ├── ingestion/                 # Data ingestion scripts
│   │   ├── download_dataset.py    # Download from Zenodo
│   │   └── __init__.py
│   ├── etl/                       # ETL pipeline scripts
│   │   ├── preprocess.py          # Image validation & normalization
│   │   └── __init__.py
│   ├── features/                  # Feature extraction scripts
│   │   └── __init__.py
│   └── catalog/                   # Data catalog
│       ├── catalog.py
│       └── __init__.py
│
├── notebooks/                     # Jupyter notebooks
│   ├── EDA.ipynb                  # Exploratory Data Analysis
│   ├── ETL.ipynb                  # ETL pipeline notebook
│   └── .ipynb_aml_checkpoints/    # Auto-saved checkpoints
│
├── pipelines/                     # Azure ML pipeline YAMLs
│   ├── training_pipeline.yml      # Main training job definition
│   ├── labeled_dataset.yml        # Dataset registration
│   ├── cnn_training_pipeline.yml  # CNN training job
│   └── backup/                    # Archived test pipelines
│
├── azure/                         # Azure ML configuration
│   ├── aml_environment.yml        # Environment definition (v2)
│   ├── batch_endpoint.yml         # Batch endpoint config
│   ├── batch_deployment.yml       # Deployment config
│   ├── model_registration.yml     # Model registration template
│   ├── catalog.json               # Data catalog metadata
│   └── score.py                   # Scoring script copy
│
├── data/                          # Data files (not in Git)
│   ├── raw/                       # Original magnetogram images
│   │   └── Lat60_Lon60_Nans0_png_224/
│   │       ├── 1065/              # Active Region 1065 images
│   │       ├── 1072/              # Active Region 1072 images
│   │       └── ...                # 1570 AR folders
│   ├── dryad/                     # Dryad label files
│   │   ├── C1.0_24hr_224_png_Labels.txt
│   │   ├── Train_Data_by_AR_png_224.csv
│   │   ├── Test_Data_by_AR_png_224.csv
│   │   ├── Validation_Data_by_AR_png_224.csv
│   │   └── Lat60_Lon60_Nans0_C1.0_24hr_png_224_features.csv
│   ├── features_labeled.csv       # Final labeled dataset (551 MB)
│   └── sample_5000_balanced.csv   # Balanced sample for testing
│
├── outputs/                       # Model outputs (not in Git)
│   ├── cnn/                       # CNN model artifacts
│   │   ├── cnn_best.keras         # Best model (25 MB)
│   │   └── cnn_model.keras        # Final model (25 MB)
│   ├── backup/                    # Archived outputs
│   │   └── features_phase1_unlabeled.csv
│   ├── features_with_metadata.csv # Phase 1 features with metadata
│   ├── features_with_target.csv   # Phase 1 features (unlabeled)
│   └── validation_report.csv      # ETL validation report
│
├── test_images/                   # Test images for deployment
│   └── 1072_hmi.M_720s.*.png      # Sample images from AR 1072
│
└── check_data.py                  # Quick data validation script
```

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

#### Model Development

Two models were developed to test the hypothesis:

**Baseline Model: Logistic Regression**
- Trained on extracted features (31 statistical, gradient, and spatial features)
- Reproducible with random seed 42 and stratified 80/20 split
- Serves as performance baseline for comparison

**Full-Feature Model: CNN (MobileNetV2)**
- Architecture: MobileNetV2 backbone pretrained on ImageNet
- Fine-tuning: Last 30 layers unfrozen for domain adaptation
- Input: 224×224 RGB magnetogram images
- Output: Binary classification (flare vs no-flare)
- Training: 10 epochs with early stopping (patience=3)
- Class weights: Balanced to handle class imbalance

#### Model Validation

**Validation Strategy:**
- Stratified 80/20 train/test split (maintaining class distribution)
- Random seed fixed at 42 for reproducibility
- Evaluation metrics: Accuracy, F1-Score, ROC-AUC, Confusion Matrix

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

### Model Versioning and Registration

The CNN model is registered in Azure ML with comprehensive metadata:

```yaml
name: solar-flare-cnn
version: 1
description: CNN model for solar flare prediction (MobileNetV2, AUC=0.9999)
properties:
  model_type: "CNN_MobileNetV2"
  framework: "TensorFlow"
  val_auc: "0.9999"
  val_accuracy: "0.94"
  img_size: "224x224"
tags:
  project: "solar-flare-prediction"
  phase: "2"
```

Registration Command:

```bash
az ml model create --name solar-flare-cnn --version 1 --path outputs/cnn/cnn_best.keras --type custom_model
```

## Deployment

The model is deployed as a **batch endpoint** on Azure Machine Learning.

### Endpoint Configuration

```yaml
name: solar-flare-batch-endpoint
auth_mode: aad_token
```

### Deployment Configuration:

```yaml
name: solar-flare-cnn-deployment
model: azureml:solar-flare-cnn:1
scoring_script: score.py
environment: azureml:solar-flare-env:2
compute: azureml:solar-flare-cluster
```

## Scoring Script Features:
init(): Loads model from registered path
run(mini_batch): Processes batch of images, returns predictions
Input: List of PNG image file paths
Output: DataFrame with filename, prediction, confidence

## Deployment Validation
The deployed endpoint was tested with 268 images from Active Region 1072:

Test Metric	Result
|----------------|--------------------|
|Input Images |	268 |
|Successful Predictions |	268 |
|Processing Time |	< 30 seconds |
|Output Format |	CSV with filename, prediction, confidence |


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
