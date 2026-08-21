#!/usr/bin/env python3
"""Push/inspect the Khidmeti STT fine-tune kernel on Kaggle (P4c). Stdlib only.
Sibling of build_push.py — GPU T4, internet ON, dataset = HF Casablanca (no payload).
KGAT token works ONLY as `Authorization: Bearer` (basic auth 401s).
Usage: stt_push.py push | status | log
"""
import json, os, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # khid-back/
ENV = dict(l.strip().split("=", 1) for l in (ROOT / ".env").read_text().splitlines()
           if "=" in l and not l.strip().startswith("#"))
# Override credentials par env SANS toucher .env (ex. compte d'un collègue) :
# KAGGLE_USERNAME=walidfg KAGGLE_KEY=… python3 stt_push.py push …
USER = os.environ.get("KAGGLE_USERNAME") or ENV["KAGGLE_USERNAME"]
KEY  = os.environ.get("KAGGLE_KEY") or ENV["KAGGLE_KEY"]
# HF_TOKEN suit le même principe : override si fourni, sinon .env.
HFTOK = os.environ.get("HF_TOKEN") or ENV["HF_TOKEN"]
SLUG = "khidmeti-stt-train"
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
    kernel = sys.argv[2] if len(sys.argv) > 2 else "stt_kernel.py"
    slug   = sys.argv[3] if len(sys.argv) > 3 else SLUG
    ksrc   = sys.argv[4:]                     # kernels montés en source de données
    src = (HERE / kernel).read_text() \
        .replace("{{HF_TOKEN}}", HFTOK) \
        .replace("{{KAGGLE_USERNAME}}", USER)
    if "{{LM_DOMAINE}}" in src:               # corpus domaine embarqué (LM kernel)
        src = src.replace("{{LM_DOMAINE}}",
                          (HERE / "lm_domaine.txt").read_text(encoding="utf-8"))
    assert "{{" not in src, "unresolved placeholder"
    body = {
        "slug": f"{USER}/{slug}", "newTitle": slug.replace("-", " ").title(),
        "text": src, "language": "python", "kernelType": "script",
        "isPrivate": True, "enableGpu": True, "enableTpu": False, "enableInternet": True,
        "machineShape": "NvidiaTeslaT4",  # P100 crashes: Kaggle torch dropped sm_60
        # DATASET=user/slug monte un dataset privé dans /kaggle/input (ex. v10) —
        # forme chaînes, identique à build_push_v6.py (recette qui tourne)
        "datasetDataSources": [os.environ["DATASET"]]
                              if os.environ.get("DATASET") else [],
        "competitionDataSources": [],
        "kernelDataSources": ksrc,
    }
    if os.environ.get("CPU") == "1":      # décodage/LM : pas un gramme de GPU (quota T4 gardé)
        body["enableGpu"] = False
        del body["machineShape"]
    print(json.dumps(call("/kernels/push", body), indent=2))

def status():
    slug = sys.argv[2] if len(sys.argv) > 2 else SLUG
    print(json.dumps(call(f"/kernels/status?userName={USER}&kernelSlug={slug}")))

def cancel():
    # Annule le run EN COURS d'un kernel (sa version relancée prendra la main).
    # Body = {"id": kernelId, "userName": …, "kernelSlug": …} — l'id seul suffit
    # (celui du dernier push), le slug identifie la version à re-run après.
    slug = sys.argv[2] if len(sys.argv) > 2 else SLUG
    rid  = sys.argv[3] if len(sys.argv) > 3 else None
    body = {"userName": USER, "kernelSlug": slug}
    if rid:
        body["id"] = int(rid)
    print(json.dumps(call("/kernels/cancel", body)))

def log():
    slug = sys.argv[2] if len(sys.argv) > 2 else SLUG
    out = call(f"/kernels/output?userName={USER}&kernelSlug={slug}")
    raw = out.get("log") or "[]"
    entries = json.loads(raw) if isinstance(raw, str) else raw
    for e in entries:
        sys.stdout.write(e.get("data", ""))

{"push": push, "status": status, "log": log, "cancel": cancel}[sys.argv[1]]()
