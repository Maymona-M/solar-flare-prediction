"""
Data catalog: Registers dataset schema, metadata, and lineage
for the solar flare magnetogram dataset.
"""

import os
import json
from datetime import datetime

# ── Catalog Definition ────────────────────────────────────────────────────────

CATALOG = {
    "dataset_name": "Solar Flare Magnetogram Dataset",
    "version": "1.0",
    "created": datetime.now(datetime.UTC).isoformat(),
    "source": {
        "url": "https://zenodo.org/records/7775776",
        "doi": "10.5281/zenodo.7775776",
        "license": "Creative Commons Attribution 4.0 International",
        "download_date": datetime.now(datetime.UTC).strftime("%Y-%m-%d")
    },
    "description": (
        "Reduced resolution (224x224 PNG) active region magnetograms "
        "from NASA Solar Dynamics Observatory (SDO), used for solar "
        "flare prediction. Labels sourced from companion Dryad dataset."
    ),
    "storage": {
        "account": "solarflarestorageproject",
        "containers": {
            "raw":       "raw/magnetograms/   — original PNG images as downloaded",
            "processed": "processed/clean/    — validated, grayscale 224x224 PNGs",
            "curated":   "curated/features_v1/ — feature-engineered dataset"
        }
    },
    "schema": {
        "images": {
            "format":     "PNG",
            "resolution": "224x224 pixels",
            "channels":   "Grayscale (1 channel)",
            "dtype":      "uint8"
        },
        "labels": {
            "source":      "https://doi.org/10.5061/dryad.jq2bvq898",
            "target":      "solar flare occurrence",
            "type":        "binary classification (flare / no-flare)",
            "label_column": "flare_class"
        }
    },
    "lineage": [
        {
            "step":   "ingestion",
            "script": "src/ingestion/download_dataset.py",
            "input":  "Zenodo record 7775776",
            "output": "raw/magnetograms/"
        },
        {
            "step":   "etl",
            "script": "src/etl/preprocess.py",
            "input":  "raw/magnetograms/",
            "output": "processed/clean/"
        },
        {
            "step":   "features",
            "script": "src/features/extract_features.py",
            "input":  "processed/clean/",
            "output": "curated/features_v1/"
        }
    ],
    "assumptions": [
        "Images are already 224x224 — no resizing needed",
        "Grayscale conversion applied during ETL",
        "Blank images (std < 1e-5) are flagged as invalid",
        "Labels joined on filename from Dryad metadata CSV"
    ]
}


def save_catalog(output_path: str = "azure/catalog.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(CATALOG, f, indent=2)
    print(f"Catalog saved to {output_path}")


if __name__ == "__main__":
    save_catalog()
    print(json.dumps(CATALOG, indent=2))