"""
ETL pipeline: Validates, cleans and preprocesses solar flare magnetogram images.
Reads from raw container, writes to processed container.
Includes resume capability and corrupted image handling.
"""

import os
import pandas as pd
from PIL import Image
import numpy as np
from io import BytesIO
from azure.storage.blob import BlobServiceClient
from tqdm import tqdm

# Configuration
STORAGE_ACCOUNT = "solarflarestorageproject"
STORAGE_KEY     = os.environ.get("AZURE_STORAGE_KEY")
RAW_CONTAINER   = "raw"
PROC_CONTAINER  = "processed"
IMAGE_SIZE      = (224, 224)


def get_client(container: str):
    client = BlobServiceClient(
        account_url=f"https://{STORAGE_ACCOUNT}.blob.core.windows.net",
        credential=STORAGE_KEY
    )
    return client.get_container_client(container)


def validate_image(img: Image.Image, filename: str) -> dict:
    """Run basic quality checks on an image."""
    arr = np.array(img)
    return {
        "filename":   filename,
        "width":      img.width,
        "height":     img.height,
        "mode":       img.mode,
        "has_nan":    bool(np.isnan(arr).any()),
        "is_blank":   bool(arr.std() < 1e-5),
        "min_pixel":  float(arr.min()),
        "max_pixel":  float(arr.max()),
        "mean_pixel": float(arr.mean()),
        "valid":      img.size == IMAGE_SIZE and not np.isnan(arr).any()
    }


def process_and_upload(raw_client, proc_client) -> pd.DataFrame:
    """Download each image from raw, validate, and upload to processed.
    Skips corrupted images and resumes from where it left off.
    """
    blobs = list(raw_client.list_blobs(name_starts_with="magnetograms/"))
    print(f"Found {len(blobs)} blobs in raw container.")

    # Resume capability: skip already processed blobs
    processed = set(
        b.name.replace("clean/", "magnetograms/")
        for b in proc_client.list_blobs(name_starts_with="clean/")
    )
    remaining = [b for b in blobs if b.name not in processed]
    print(f"Already processed: {len(processed)}. Remaining: {len(remaining)}.")

    report = []
    for blob in tqdm(remaining):
        try:
            # Download
            data = raw_client.download_blob(blob.name).readall()

            # Skip suspiciously small files
            if len(data) < 100:
                print(f"Skipping too-small file: {blob.name}")
                continue

            # Verify image integrity before processing
            try:
                img = Image.open(BytesIO(data))
                img.verify()
                img = Image.open(BytesIO(data)).convert("L")  # reopen after verify
            except Exception:
                print(f"Skipping corrupted image: {blob.name}")
                continue

            stats = validate_image(img, blob.name)
            report.append(stats)

            if not stats["valid"]:
                continue

            # Upload to processed container
            proc_client.upload_blob(
                name=blob.name.replace("magnetograms/", "clean/"),
                data=data,
                overwrite=True
            )

        except Exception as e:
            print(f"Skipping {blob.name}: {e}")
            continue

    return pd.DataFrame(report)


if __name__ == "__main__":
    if not STORAGE_KEY:
        raise ValueError("Set AZURE_STORAGE_KEY environment variable.")

    raw_client  = get_client(RAW_CONTAINER)
    proc_client = get_client(PROC_CONTAINER)

    print("Starting ETL pipeline...")
    report_df = process_and_upload(raw_client, proc_client)

    os.makedirs("outputs", exist_ok=True)
    report_df.to_csv("outputs/validation_report.csv", index=False)

    print(f"\nETL complete.")
    if len(report_df) > 0:
        print(f"Valid images:   {report_df['valid'].sum()}")
        print(f"Invalid images: {(~report_df['valid']).sum()}")
    print(f"Report saved to outputs/validation_report.csv")