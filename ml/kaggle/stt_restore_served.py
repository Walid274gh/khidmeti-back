#!/usr/bin/env python3
"""Restaure l'état PRODUCTION connu-bon du STT CTC (CPU, transport HF).

CONTEXTE (Aug 20 2026) : le bac à sable `khidmeti-stt-ctc-cand` a été pollué
par la suite 93h — model.int8.onnx écrasé par le modèle 93h B_ep3 (PIRE :
greedy 0.6798 vs 0.6688 du 24h), puis lm.npz remplacé par un LM order-4
MÉLANGÉ que le serveur de prod ne sait pas charger (assert order==3 → crash
au boot → conteneur en boucle → ai-stt injoignable → FALLBACK → 0 %).

L'état MESURÉ bon (WER 0,6236) est le commit 63c09cef1612 (11:22, run LM sur
les poids 24h B_ep8 + domaine 214) :
  model.int8.onnx  = 356 359 085 o   ← déposé par le train-24h à 00:38, jamais
                                        retouché par les runs LM (ils n'écrivent
                                        que lm.npz/meta.json/hyps.json)
  ctc_vocab.json   = 193 tokens (blank 64, delim "|")
  lm.npz           = LM écrit order-3 + domaine 214 → 2 293 561 ngrams
  meta.json        = v10-ctc-lm, lm.order=3 is_mix=false w_ecrit=1.0,
                     alpha=0.3 beta=2.0 floor=-30.0 beam=128

Ce kernel copie ces 4 fichiers vers le repo PRIVÉ `Walidrbh27/khidmeti-stt-ctc`
— le repo que le gate du kernel LM devait écrire en cas de succès, désormais
le repo SERVÉ en prod. `-cand` redevient un bac à sable pour les runs.
"""
import json, os
from huggingface_hub import hf_hub_download, HfApi

HF_TOKEN = "{{HF_TOKEN}}"
os.environ["HF_TOKEN"] = HF_TOKEN

CAND    = "Walidrbh27/khidmeti-stt-ctc-cand"
REV     = "63c09cef1612"          # état connu-bon (24h B_ep8 + LM order-3 domaine)
REPO    = "Walidrbh27/khidmeti-stt-ctc"     # repo SERVÉ (celui du gate-pass)
FILES   = ["model.int8.onnx", "ctc_vocab.json", "lm.npz", "meta.json"]

api = HfApi(token=HF_TOKEN)
api.create_repo(REPO, private=True, exist_ok=True)

for name in FILES:
    local = hf_hub_download(CAND, name, revision=REV, token=HF_TOKEN)
    size = os.path.getsize(local)
    api.upload_file(path_or_fileobj=local, path_in_repo=name, repo_id=REPO)
    print(f"RESTORE_COPIED {name:18s} {size:>12d} o", flush=True)

# ── vérification à froid : on relit CE QU'ON A ÉCRIT, pas ce qu'on a envoyé ──
meta = json.load(open(hf_hub_download(REPO, "meta.json", token=HF_TOKEN),
                       encoding="utf-8"))
lm = meta.get("lm") or {}
print("RESTORE_META", meta.get("version"), "serve_file=", meta.get("serve_file"),
      "| lm.order=", lm.get("order"), "is_mix=", lm.get("is_mix"),
      "w_ecrit=", lm.get("w_ecrit"), "ngrams=", lm.get("ngrams"),
      "alpha=", lm.get("alpha"), "beta=", lm.get("beta"),
      "beam=", lm.get("beam"), flush=True)
assert meta.get("version") == "v10-ctc-lm", meta.get("version")
assert lm.get("is_mix") is False and lm.get("order") == 3, lm
assert meta.get("serve_file") == "model.int8.onnx", meta.get("serve_file")

print("RESTORE_DONE " + REPO, flush=True)