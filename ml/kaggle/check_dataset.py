#!/usr/bin/env python3
"""Upload dataset to Kaggle using direct API calls (no kaggle CLI needed)."""
import json, sys, urllib.request, urllib.parse, base64, zipfile, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENV = dict(l.strip().split("=", 1) for l in (ROOT / ".env").read_text().splitlines()
           if "=" in l and not l.strip().startswith("#"))

USER = ENV["KAGGLE_USERNAME"]
KEY = ENV["KAGGLE_KEY"]
API = "https://www.kaggle.com/api/v1"
DATASET_SLUG = f"{USER}/khidmeti-nlu-v6-data"

def call(path, body=None, method=None):
    """Make authenticated API call to Kaggle."""
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode() if body else None,
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json"
        },
        method=method or ("POST" if body else "GET")
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            resp = r.read().decode()
            return json.loads(resp) if resp else {}
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        print(f"HTTP {e.code}: {err}", file=sys.stderr)
        return None

# Step 1: Check if dataset exists
print(f"Checking if dataset exists: {DATASET_SLUG}")
existing = call(f"/datasets/view/{DATASET_SLUG}")

if existing:
    print("✅ Dataset already exists!")
    print(f"   URL: https://www.kaggle.com/datasets/{DATASET_SLUG}")
    print(f"   Files: {[f['name'] for f in existing.get('datasetFiles', [])]}")
    print("\nDataset is ready to use in kernel.")
    sys.exit(0)
else:
    print("❌ Dataset doesn't exist yet.")
    print("\nKaggle API doesn't support programmatic dataset creation.")
    print("You need to create it manually once:")
    print()
    print("=" * 80)
    print("MANUAL STEPS:")
    print("=" * 80)
    print(f"1. Go to: https://www.kaggle.com/datasets")
    print(f"2. Click 'New Dataset'")
    print(f"3. Upload file: {HERE / 'nlu_v6_data.zip'}")
    print(f"4. Title: Khidmeti NLU v6 Training Data")
    print(f"5. Visibility: Private")
    print(f"6. Click 'Create'")
    print()
    print("Once created, run this script again to verify.")
    print("Then run: python3 ml/kaggle/build_push_v6.py push")
    sys.exit(1)
