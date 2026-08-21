#!/usr/bin/env python3
"""Push/inspect the Khidmeti SigLIP vision-export kernel on Kaggle (P5).
Sibling of build_push.py (NLU) — CPU kernel, no GPU, internet ON.
KGAT token works ONLY as `Authorization: Bearer` (basic auth 401s).
Usage: vision_push.py push | status | log
"""
import base64, gzip, json, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # khid-back/
ENV = dict(l.strip().split("=", 1) for l in (ROOT / ".env").read_text().splitlines()
           if "=" in l and not l.strip().startswith("#"))
USER, KEY = ENV["KAGGLE_USERNAME"], ENV["KAGGLE_KEY"]
SLUG = "khidmeti-vision-export"
API = "https://www.kaggle.com/api/v1"

def call(path, body=None):
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode() if body else None,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        method="POST" if body else "GET")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()[:2000]}", file=sys.stderr)
        sys.exit(1)

def push():
    src = (HERE / "vision_kernel.py").read_text()
    labels = base64.b64encode(gzip.compress(
        (ROOT / "ml" / "dataset" / "labels.json").read_bytes(), 9)).decode()
    src = src.replace("{{HF_TOKEN}}", ENV["HF_TOKEN"]).replace("{{LABELS_B64}}", labels)
    assert "{{" not in src, "unresolved placeholder"
    print(json.dumps(call("/kernels/push", {
        "slug": f"{USER}/{SLUG}", "newTitle": "Khidmeti vision export",
        "text": src, "language": "python", "kernelType": "script",
        "isPrivate": True, "enableGpu": False, "enableTpu": False, "enableInternet": True,
        "datasetDataSources": [], "competitionDataSources": [], "kernelDataSources": [],
    }), indent=2))

def status():
    print(json.dumps(call(f"/kernels/status?userName={USER}&kernelSlug={SLUG}")))

def log():
    out = call(f"/kernels/output?userName={USER}&kernelSlug={SLUG}")
    raw = out.get("log") or "[]"
    entries = json.loads(raw) if isinstance(raw, str) else raw
    for e in entries:
        sys.stdout.write(e.get("data", ""))

{"push": push, "status": status, "log": log}[sys.argv[1]]()
