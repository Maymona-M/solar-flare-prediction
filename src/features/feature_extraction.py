"""
Feature extraction for solar flare magnetogram images.
Computes statistical, gradient, and texture features from each image.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from scipy import stats
from skimage.feature import grayscale_matrix  
from azure.storage.blob import BlobServiceClient
from tqdm import tqdm


# Configuration
STORAGE_ACCOUNT = "solarflarestorageproject"
STORAGE_KEY = os.environ.get("AZURE_STORAGE_KEY", "")
PROC_CONTAINER = "processed"


def get_container_client(container):
    client = BlobServiceClient(
        account_url=f"https://{STORAGE_ACCOUNT}.blob.core.windows.net",
        credential=STORAGE_KEY
    )
    return client.get_container_client(container)


def parse_label(blob_name):
    """Extract flare class from filename."""
    filename = blob_name.split("/")[-1]
    for cls in ["X", "M", "C", "B"]:
        if filename.startswith(cls):
            return cls
    return "N"


def extract_statistical_features(arr):
    """Compute pixel-level statistical features."""
    flat = arr.flatten()
    return {
        "mean":     float(np.mean(flat)),
        "std":      float(np.std(flat)),
        "min":      float(np.min(flat)),
        "max":      float(np.max(flat)),
        "skewness": float(stats.skew(flat)),
        "kurtosis": float(stats.kurtosis(flat)),
        "p25":      float(np.percentile(flat, 25)),
        "p75":      float(np.percentile(flat, 75)),
        "iqr":      float(np.percentile(flat, 75) - np.percentile(flat, 25)),
    }


def extract_gradient_features(arr):
    """Compute gradient-based edge features."""
    gy, gx = np.gradient(arr.astype(np.float32))
    magnitude = np.sqrt(gx**2 + gy**2)
    return {
        "grad_mean": float(np.mean(magnitude)),
        "grad_std":  float(np.std(magnitude)),
        "grad_max":  float(np.max(magnitude)),
    }


def extract_spatial_features(arr):
    """Compute spatial distribution features."""
    h, w = arr.shape
    total = arr.sum()
    if total == 0:
        cx, cy = w / 2, h / 2
    else:
        cx = (arr * np.arange(w)).sum() / total
        cy = (arr.T * np.arange(h)).sum() / total
    threshold = arr.mean()
    active_pixels = (arr > threshold).sum()
    return {
        "center_x":     float(cx),
        "center_y":     float(cy),
        "active_ratio": float(active_pixels / (h * w)),
    }


def extract_features(img, blob_name):
    """Extract all features from a single image."""
    arr = np.array(img).astype(np.float32)
    features = {"filename": blob_name, "label": parse_label(blob_name)}
    features.update(extract_statistical_features(arr))
    features.update(extract_gradient_features(arr))
    features.update(extract_spatial_features(arr))
    return features


def run_feature_extraction(container_client, max_images=None):
    """Run feature extraction on all images in processed container."""
    blobs = list(container_client.list_blobs(name_starts_with="clean/"))
    if max_images:
        blobs = blobs[:max_images]

    print(f"Extracting features from {len(blobs)} images...")
    records = []

    for blob in tqdm(blobs):
        try:
            data = container_client.download_blob(blob.name).readall()
            img = Image.open(BytesIO(data)).convert("L")
            features = extract_features(img, blob.name)
            records.append(features)
        except Exception as e:
            print(f"Error processing {blob.name}: {e}")
            continue

    return pd.DataFrame(records)


if __name__ == "__main__":
    if not STORAGE_KEY:
        raise ValueError("Set AZURE_STORAGE_KEY environment variable.")

    proc_client = get_container_client(PROC_CONTAINER)
    features_df = run_feature_extraction(proc_client)

    if len(features_df) > 0:
        os.makedirs("outputs", exist_ok=True)
        features_df.to_csv("outputs/features.csv", index=False)
        print(f"Feature extraction complete.")
        print(f"Total images: {len(features_df)}")
        print(f"Features per image: {len(features_df.columns) - 2}")
        print(features_df.head())
    else:
        print("No images found. Run after ETL pipeline completes.")