#!/usr/bin/env python3
"""
Upload Khidmeti NLU v6 dataset to Kaggle, then reference it in kernel.
Strategy:
  1. Create/update dataset with synth_v6_final.csv + eval_heldout.csv + labels.json
  2. Update train_kernel.py to load from Kaggle dataset instead of inline b64
  3. Push kernel with dataset reference
"""
import json, sys, urllib.request, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENV = dict(l.strip().split("=", 1) for l in (ROOT / ".env").read_text().splitlines()
           if "=" in l and not l.strip().startswith("#"))
USER, KEY = ENV["KAGGLE_USERNAME"], ENV["KAGGLE_KEY"]
API = "https://www.kaggle.com/api/v1"
DATASET_SLUG = f"{USER}/khidmeti-nlu-v6-data"

def call(path, body=None, method=None):
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode() if body else None,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        method=method or ("POST" if body else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            resp = r.read().decode()
            return json.loads(resp) if resp else {}
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()[:2000]}", file=sys.stderr)
        if e.code == 404 and "create" in path:  # dataset doesn't exist yet
            return None
        sys.exit(1)

def create_or_update_dataset():
    """Upload dataset files to Kaggle."""
    ds_dir = ROOT / "ml" / "dataset"

    # Create metadata
    metadata = {
        "title": "Khidmeti NLU v6 Training Data",
        "id": DATASET_SLUG,
        "licenses": [{"name": "CC0-1.0"}],
        "resources": [
            {
                "path": "synth_v6_final.csv",
                "description": "70k training rows (10.7x v5) with augmentations"
            },
            {
                "path": "eval_heldout.csv",
                "description": "Heldout evaluation set (never touched during dev)"
            },
            {
                "path": "labels.json",
                "description": "Intent and profession labels (frozen)"
            }
        ]
    }

    # Create zip with files
    zip_path = HERE / "nlu_v6_data.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(ds_dir / "synth_v6_final.csv", "synth_v6_final.csv")
        zf.write(ds_dir / "eval_heldout.csv", "eval_heldout.csv")
        zf.write(ds_dir / "labels.json", "labels.json")
        zf.writestr("dataset-metadata.json", json.dumps(metadata, indent=2))

    print(f"Created zip: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Check if dataset exists
    try:
        existing = call(f"/datasets/view/{DATASET_SLUG}")
        action = "update"
        print(f"Dataset exists, will update: {DATASET_SLUG}")
    except:
        action = "create"
        print(f"Dataset doesn't exist, will create: {DATASET_SLUG}")

    # Upload (Kaggle API requires using kaggle CLI or manual upload for now)
    print("\n" + "=" * 80)
    print("MANUAL STEP REQUIRED:")
    print("=" * 80)
    print(f"\n1. Go to: https://www.kaggle.com/datasets")
    print(f"2. Click 'New Dataset'")
    print(f"3. Upload: {zip_path}")
    print(f"4. Set title: Khidmeti NLU v6 Training Data")
    print(f"5. Set visibility: Private")
    print(f"6. Click 'Create'")
    print(f"\nOnce created, the dataset will be at:")
    print(f"   https://www.kaggle.com/datasets/{DATASET_SLUG}")
    print(f"\nDataset slug for kernel: {DATASET_SLUG}")
    print("\nAfter upload, run: python3 ml/kaggle/build_push_v6.py push")

    return zip_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 prepare_dataset.py create")
        sys.exit(1)

    if sys.argv[1] == "create":
        create_or_update_dataset()
    else:
        print(f"Unknown command: {sys.argv[1]}")
        sys.exit(1)
