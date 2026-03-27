# Solar Flare Prediction

**Course:** DSAI 3202 — Data Pipeline, ETL, and Feature Foundations  
**Institution:** University of Doha for Science and Technology

---

## Team

| Name | Student ID | Role |
|---|---|---|
| Maymona Mustafa | 60306027 | Model Development, Versioning and Deployment  |
| Asma Riyaz | 60305750 | Model Development & Validation |

---

## Project Description

This project develops a supervised machine learning system to predict solar flare occurrence using magnetogram images of solar active regions. The system ingests large-scale image data from a publicly accessible scientific repository, preprocesses and normalizes the images, and extracts features for use in downstream classification models.

**Hypothesis:** A supervised model trained on full-resolution solar magnetogram images will predict solar flare occurrence more accurately than a baseline model trained on reduced or simplified image features.

**Dataset:** Active Region Magnetograms for Solar Flare Prediction (Zenodo, NASA SDO/HMI)  
**Source:** https://doi.org/10.5281/zenodo.7775776  
**Size:** ~13 GB · ~950,000 PNG images · 224×224 resolution

---

## Phase 2 Objectives

Phase 2 focuses on model development, validation, and deployment within an AI system.

1. **Model Development** — Select appropriate model(s) aligned with the project hypothesis and data characteristics. Establish at least one baseline model. Training must be reproducible with explicit handling of data splits, parameters, and random seeds.
2. **Model Validation** — Define and apply a validation strategy appropriate to the problem context, including selection of evaluation metrics, error analysis, and comparison against baselines. Validation must reflect realistic usage conditions and avoid data leakage.
3. **Model Versioning and Registration** — Version trained models and associated artifacts. Register models with clear metadata including training data version, feature set, metrics, and limitations. Ensure traceability between data, code, and deployed models.
4. **Deployment** — Deploy the selected model using an appropriate serving mode (batch or real-time). Define clear input and output interfaces, ensure feature parity between training and serving, and document deployment configuration.
5. **Deployment Validation** — Verify deployed model behavior through functional tests and sanity checks. Confirm correctness, latency expectations, and consistency between offline and deployed predictions.

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

## Phase 2 Implementation

### Labeled Dataset Preparation

The Phase 1 features were merged with Dryad labels to create a properly labeled dataset for training:

| Source | Description |
|--------|-------------|
| Features | `Lat60_Lon60_Nans0_C1.0_24hr_png_224_features.csv` (950,047 rows) |
| Labels | `C1.0_24hr_224_png_Labels.txt` from Dryad (filename, flare_class) |
| Merge Strategy | Filename-based join using base filename (e.g., `1064_hmi.M_720s.20100501_000000_TAI.1.magnetogram_224.png`) |
| Output | `features_labeled.csv` (950,047 rows, 18 features + flare_class + binary target) |

**Class Distribution:**
- Flare positive (≥C1.0): 878,485 images (92.5%)
- Flare negative: 71,562 images (7.5%)

#### Compute Configuration

```bash
# Compute cluster created with:
az ml compute create --name solar-flare-cluster \
  --type AmlCompute \
  --size Standard_DS3_v2 \
  --min-instances 0 \
  --max-instances 2
```
#### Environment Configuration

```bash
# azure/aml_environment.yml
name: solar-flare-env
version: 1
dependencies:
  - python=3.9
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - Pillow
  - pip
  - pip:
    - torch==2.0.1
    - torchvision==0.15.2
    - tensorflow
    - azureml-mlflow
```

#### Dataset Registration

The labeled dataset is registered in Azure ML for versioned access:

```bash
# Dataset registration YAML
# pipelines/labeled_dataset.yml
az ml data create -f pipelines/labeled_dataset.yml
```

#### Model Training Pipeline

The training pipeline is defined in pipelines/training_pipeline.yml and accepts the labeled dataset as input.

```bash
# Submit training job
az ml job create -f pipelines/training_pipeline.yml \
  --resource-group rg-60306027 \
  --workspace-name solar-flare-aml-60306027
```

#### Batch Deployment Configuration

The batch endpoint is configured for model serving:

```bash
# azure/batch_endpoint.yml
name: solar-flare-batch-endpoint
description: Batch endpoint for solar flare prediction
auth_mode: aad_token

# azure/batch_deployment.yml
name: solar-flare-model-deployment
endpoint_name: solar-flare-batch-endpoint
model: azureml:solar-flare-model:1
code_configuration:
  code: ../src
  scoring_script: score.py
environment: azureml:solar-flare-env:1
compute: azureml:solar-flare-cluster
```

#### Scoring Script

The scoring script src/score.py implements:

```bash init():``` Loads the trained model from Azure ML

```bash run(mini_batch):``` Processes batches of images and returns predictions

Input: List of PNG image file paths
Output: DataFrame with columns: ```bash filename ```, ```bash prediction ```, ```bash confidence ```

---

## Azure Infrastructure

| Resource | Name | Description |
|----------|------|-------------|
| Resource Group | `rg-60306027` | Container for all Azure resources |
| Storage Account | `solarflarestorageproject` | Raw, processed, and curated image storage |
| AML Workspace | `solar-flare-aml-60306027` | Central ML resource management |
| Compute Cluster | `solar-flare-cluster` | Standard_DS3_v2, 0-2 nodes, auto-scale |
| Environment | `solar-flare-env:1` | PyTorch 2.0.1, scikit-learn, pandas, tensorflow |
| Labeled Dataset | `solar-flare-labeled:1` | Versioned dataset with 950,047 labeled samples |

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Maymona-M/solar-flare-prediction.git
cd solar-flare-prediction
git checkout phase1
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
# Ingest data
python src/ingestion/download_dataset.py

# Run ETL
python src/etl/preprocess.py

# Extract features
python src/features/feature_extraction.py
```

---

## Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, submitted snapshots |
| `phase1` | Phase 1 development (data foundation) |
| `phase2` | Phase 2 development (modeling & deployment) - current |

---

## Citation

Boucheron, L. E., Vincent, T., Grajeda, J. A., & Wuest, E. (2023). Active Region Magnetograms for Solar Flare Prediction: Reduced Resolution Dataset Images. Zenodo. https://doi.org/10.5281/zenodo.7775776
