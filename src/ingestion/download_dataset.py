"""
Ingestion script: Downloads solar flare magnetogram dataset from Zenodo
and uploads to Azure Blob Storage (raw container).
"""

import os
import subprocess
from azure.storage.blob import BlobServiceClient
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
ZENODO_RECORD_ID  = "7775776"
DOWNLOAD_DIR      = "data/raw"
STORAGE_ACCOUNT   = "solarflarestorageproject"
CONTAINER_NAME    = "raw"
STORAGE_KEY       = os.environ.get("AZURE_STORAGE_KEY")  # never hardcode this


def download_from_zenodo(record_id: str, output_dir: str) -> str:
    """Download dataset from Zenodo using zenodo_get."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading Zenodo record {record_id} to {output_dir}...")
    subprocess.run(
        ["zenodo_get", record_id, "-o", output_dir],
        check=True
    )
    # Find the downloaded tar file
    for f in os.listdir(output_dir):
        if f.endswith(".tar.gz"):
            return os.path.join(output_dir, f)
    raise FileNotFoundError("No .tar.gz file found after download.")


def extract_archive(archive_path: str, output_dir: str) -> str:
    """Extract the tar.gz archive."""
    print(f"Extracting {archive_path}...")
    subprocess.run(
        ["tar", "-xzf", archive_path, "-C", output_dir],
        check=True
    )
    print("Extraction complete.")
    return output_dir


def upload_to_azure(local_dir: str, container: str, prefix: str = "magnetograms/"):
    """Upload all PNG files to Azure Blob Storage raw container."""
    if not STORAGE_KEY:
        raise ValueError("AZURE_STORAGE_KEY environment variable not set.")

    client = BlobServiceClient(
        account_url=f"https://{STORAGE_ACCOUNT}.blob.core.windows.net",
        credential=STORAGE_KEY
    )
    container_client = client.get_container_client(container)

    png_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(local_dir)
        for f in files if f.endswith(".png")
    ]

    print(f"Uploading {len(png_files)} PNG files to {container}/{prefix}...")
    for filepath in tqdm(png_files):
        blob_name = prefix + os.path.relpath(filepath, local_dir)
        with open(filepath, "rb") as data:
            container_client.upload_blob(
                name=blob_name,
                data=data,
                overwrite=True
            )
    print("Upload complete.")


if __name__ == "__main__":
    archive = download_from_zenodo(ZENODO_RECORD_ID, DOWNLOAD_DIR)
    extract_archive(archive, DOWNLOAD_DIR)
    upload_to_azure(DOWNLOAD_DIR, CONTAINER_NAME)