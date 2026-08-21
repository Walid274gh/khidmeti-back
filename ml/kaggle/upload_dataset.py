#!/usr/bin/env python3
"""Upload dataset to Kaggle using kaggle CLI (installed via pip)."""
import subprocess, sys, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENV = dict(l.strip().split("=", 1) for l in (ROOT / ".env").read_text().splitlines()
           if "=" in l and not l.strip().startswith("#"))

# Set Kaggle credentials as env vars
import os
os.environ["KAGGLE_USERNAME"] = ENV["KAGGLE_USERNAME"]
os.environ["KAGGLE_KEY"] = ENV["KAGGLE_KEY"]

# Install kaggle CLI if needed
try:
    import kaggle
except ImportError:
    print("Installing kaggle CLI...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)

# Create dataset metadata
DATASET_SLUG = f"{ENV['KAGGLE_USERNAME']}/khidmeti-nlu-v6-data"
metadata = {
    "title": "Khidmeti NLU v6 Training Data",
    "id": DATASET_SLUG,
    "licenses": [{"name": "CC0-1.0"}]
}

# Extract zip to temp dir
import tempfile, zipfile
with tempfile.TemporaryDirectory() as tmpdir:
    tmppath = Path(tmpdir)
    
    # Extract files
    with zipfile.ZipFile(HERE / "nlu_v6_data.zip", 'r') as zf:
        zf.extractall(tmppath)
    
    # Write metadata
    (tmppath / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))
    
    print(f"Uploading dataset to: {DATASET_SLUG}")
    
    # Use kaggle CLI
    result = subprocess.run(
        ["kaggle", "datasets", "create", "-p", str(tmppath), "--dir-mode", "zip"],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        print("✅ Dataset uploaded successfully!")
        print(result.stdout)
    else:
        # Try version update if exists
        print("Dataset may exist, trying to update version...")
        result = subprocess.run(
            ["kaggle", "datasets", "version", "-p", str(tmppath), "-m", "v6: 70k rows", "--dir-mode", "zip"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("✅ Dataset updated successfully!")
            print(result.stdout)
        else:
            print("❌ Upload failed:")
            print(result.stderr)
            sys.exit(1)

print(f"\nDataset URL: https://www.kaggle.com/datasets/{DATASET_SLUG}")
print(f"Dataset slug: {DATASET_SLUG}")
