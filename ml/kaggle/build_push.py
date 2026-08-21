#!/usr/bin/env python3
"""Push/inspect the Khidmeti NLU training kernel on Kaggle. Stdlib only.
KGAT token works ONLY as `Authorization: Bearer` (basic auth 401s).
Usage: build_push.py push | status | log
"""
import base64, gzip, json, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # khid-back/
ENV = dict(l.strip().split("=", 1) for l in (ROOT / ".env").read_text().splitlines()
           if "=" in l and not l.strip().startswith("#"))
USER, KEY = ENV["KAGGLE_USERNAME"], ENV["KAGGLE_KEY"]
SLUG = "khidmeti-nlu-train"
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

def b64(p):
    return base64.b64encode(gzip.compress(p.read_bytes(), 9)).decode()

def train_payload(ds):
    """v6_final (70k rows = 10.7× v5) — massive expansion via regional/arabizi/context.
    hand_v4 ×3 (real examples over-sampled for style transfer)."""
    synth = (ds / "synth_v6_final.csv").read_text()
    hand = (ds / "hand_v4.csv").read_text().split("\n", 1)[1].strip()
    # Triple hand-curated examples to anchor real user style
    combined = synth.rstrip("\n") + "\n" + hand + "\n" + hand + "\n" + hand + "\n"
    return base64.b64encode(gzip.compress(combined.encode(), 9)).decode()

def push():
    ds = ROOT / "ml" / "dataset"
    src = (HERE / "train_kernel.py").read_text()
    for k, v in [("HF_TOKEN", ENV["HF_TOKEN"]), ("SYNTH_B64", train_payload(ds)),
                 ("EVAL_B64", b64(ds / "eval_heldout.csv")), ("LABELS_B64", b64(ds / "labels.json"))]:
        src = src.replace("{{%s}}" % k, v)
    assert "{{" not in src, "unresolved placeholder"
    print(json.dumps(call("/kernels/push", {
        "slug": f"{USER}/{SLUG}", "newTitle": "Khidmeti NLU train",
        "text": src, "language": "python", "kernelType": "script",
        "isPrivate": True, "enableGpu": True, "enableTpu": False, "enableInternet": True,
        "machineShape": "NvidiaTeslaT4",  # P100 crashes: Kaggle torch dropped sm_60
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
