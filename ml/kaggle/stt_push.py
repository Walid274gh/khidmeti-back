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

def _read_env(p):
    try:
        return dict(l.strip().split("=", 1) for l in Path(p).read_text().splitlines()
                    if "=" in l and not l.strip().startswith("#"))
    except OSError:
        return {}

# Le `.env` de khid-back est GÉNÉRÉ (make le recopie depuis .env.cloud) : tout
# secret d'OUTILLAGE qu'on y met disparaît au prochain rebuild — c'est arrivé
# le 22/08, la clé Kaggle a été perdue en plein milieu du chantier 93 h.
# Ordre de lecture : env explicite > ~/.kaggle/kaggle.json (emplacement
# standard, hors repo) > ~/khid-secrets.env > .env (héritage, si présent).
ENV = {**_read_env(ROOT / ".env"), **_read_env(Path.home() / "khid-secrets.env")}
_KJ = Path.home() / ".kaggle" / "kaggle.json"
if _KJ.exists():
    _j = json.loads(_KJ.read_text())
    ENV.setdefault("KAGGLE_USERNAME", _j.get("username", ""))
    ENV.setdefault("KAGGLE_KEY", _j.get("key", ""))

def _need(name):
    v = os.environ.get(name) or ENV.get(name)
    if not v:
        sys.exit(f"[stt_push] {name} introuvable.\n"
                 f"  → mettez-le dans ~/khid-secrets.env (hors repo, survit au make)\n"
                 f"     ou ~/.kaggle/kaggle.json, ou passez-le en variable "
                 f"d'environnement.")
    return v

USER  = _need("KAGGLE_USERNAME")
KEY   = _need("KAGGLE_KEY")
HFTOK = _need("HF_TOKEN")
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
    # DISABLE=1 : pousse la version SANS lancer de run. C'est le SEUL frein
    # fiable — /kernels/cancel n'existe pas dans l'API publique (404 HTML), et
    # un push nu QUEUE toujours un run. Sert à figer un run en attente quand on
    # découvre un bug avant qu'il ne brûle du quota.
    if os.environ.get("DISABLE") == "1":
        body["isDisabled"] = True
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
