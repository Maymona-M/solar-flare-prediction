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
| `phase1` | Phase 1 development (current) |

---

## Citation

Boucheron, L. E., Vincent, T., Grajeda, J. A., & Wuest, E. (2023). Active Region Magnetograms for Solar Flare Prediction: Reduced Resolution Dataset Images. Zenodo. https://doi.org/10.5281/zenodo.7775776
