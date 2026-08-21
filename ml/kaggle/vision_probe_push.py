#!/usr/bin/env python3
"""Push/inspect les kernels vision CPU de Khidmeti sur Kaggle (P5b probe, P5c MC2).
Stdlib only. Sibling de vision_push.py — CPU, internet ON (collecte ddgs).
KGAT token works ONLY as `Authorization: Bearer` (basic auth 401s).
Usage: vision_probe_push.py push|status|log [kernel.py] [slug]
   ex: vision_probe_push.py push vision_kernel_mc2.py khidmeti-vision-mc2
"""
import json, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # khid-back/
ENV = dict(l.strip().split("=", 1) for l in (ROOT / ".env").read_text().splitlines()
           if "=" in l and not l.strip().startswith("#"))
USER, KEY = ENV["KAGGLE_USERNAME"], ENV["KAGGLE_KEY"]
KERNEL = "vision_probe_kernel.py"
SLUG = "khidmeti-vision-probe"
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

def _slug():
    return sys.argv[3] if len(sys.argv) > 3 else SLUG

def push():
    kernel = sys.argv[2] if len(sys.argv) > 2 else KERNEL
    slug = _slug()
    src = (HERE / kernel).read_text().replace("{{HF_TOKEN}}", ENV["HF_TOKEN"])
    assert "{{" not in src, "unresolved placeholder"
    print(json.dumps(call("/kernels/push", {
        "slug": f"{USER}/{slug}", "newTitle": slug.replace("-", " ").title(),
        "text": src, "language": "python", "kernelType": "script",
        "isPrivate": True, "enableGpu": False, "enableTpu": False, "enableInternet": True,
        "datasetDataSources": [], "competitionDataSources": [], "kernelDataSources": [],
    }), indent=2))

def status():
    print(json.dumps(call(f"/kernels/status?userName={USER}&kernelSlug={_slug()}")))

def log():
    out = call(f"/kernels/output?userName={USER}&kernelSlug={_slug()}")
    raw = out.get("log") or "[]"
    entries = json.loads(raw) if isinstance(raw, str) else raw
    for e in entries:
        sys.stdout.write(e.get("data", ""))

{"push": push, "status": status, "log": log}[sys.argv[1]]()
