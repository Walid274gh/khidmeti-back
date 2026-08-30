#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
# ml/kaggle/stt_promote_kernel_v10.py — promotion v10 : -cand → repo servi.
#
# Décision produit 26/08 : le gate « battre whisper » a échoué (faisceau+LM
# 0,6177 vs 0,6097 apparié) MAIS l'incumbent réel servi depuis le 20/08 est le
# CTC greedy à 0,6534 sur le même test — le faisceau+LM le bat de −3,6 pts WER
# et passe sous son CER whisper (0,2318 vs 0,2368), pour ~0,14 s/clip de plus.
# Le user a validé la promotion.
#
# CINQ fichiers, PAS QUATRE : le repo servi porte encore les poids v8 (int8 et
# vocab différents de ceux du run 90h). Promouvoir lm.npz seul donnerait un
# LM calibré sur d'autres logits — il faut model.int8.onnx + ctc_vocab.json +
# meta.json + lm.npz (+ lm_parle.npz pour que le chargement mix du serveur ne
# manque jamais de fichier). fp32 NON : 1,26 Go inutile en prod (serve_file=int8).
#
# UN SEUL COMMIT : upload_folder pousse les 5 fichiers ensemble — jamais de
# fenêtre « nouveaux poids + vieux vocab » sur le repo servi.
#
# Transport HF volontaire : le multipart Kaggle meurt en EOF TLS sur les gros
# fichiers (leçon du 20/08), huggingface_hub reprend les chunks.
#
#   python3 stt_push.py push stt_promote_kernel_v10.py khidmeti-stt-promote-v10
#   (CPU=1 : zéro GPU, zéro calcul ici — pur copier-coller HF)
# ══════════════════════════════════════════════════════════════════════════════
import json, shutil
from pathlib import Path

HF_TOKEN = "{{HF_TOKEN}}"
CAND   = "Walidrbh27/khidmeti-stt-ctc-cand"
SERVED = "Walidrbh27/khidmeti-stt-ctc"
FILES = ["model.int8.onnx", "ctc_vocab.json", "meta.json", "lm.npz", "lm_parle.npz"]

from huggingface_hub import hf_hub_download, HfApi

api = HfApi(token=HF_TOKEN)

out = Path("out")
out.mkdir(exist_ok=True)
for f in FILES:
    shutil.copy(hf_hub_download(CAND, f, token=HF_TOKEN), out / f)
    print(f"pull ok: {f} ({(out / f).stat().st_size:,} o)", flush=True)

# Garde-fou : le méta promu doit être LE run LM mesuré aujourd'hui, pas un vieux
# méta traînant sur -cand — sinon on sert une config dont le WER n'est pas celui
# affiché. Version attendue écrite par stt_lm_kernel_v9 (run khidmeti-stt-lm-v10).
meta = json.load(open(out / "meta.json", encoding="utf-8"))
assert meta.get("version") == "v10-ctc-lm", f"méta inattendu : {meta.get('version')!r}"
lm = meta["lm"]
assert abs(lm["alpha"] - 0.3) < 1e-9 and abs(lm["beta"] - 2.0) < 1e-9, lm
assert lm.get("beam") == 64 and lm.get("order") == 3, lm
assert abs(meta["wer_norm"]["test_int8_beam_lm"] - 0.6177) < 5e-4, meta["wer_norm"]
print("méta vérifié :", meta["version"],
      "| beam_lm test =", meta["wer_norm"]["test_int8_beam_lm"], flush=True)

api.upload_folder(folder_path=str(out), repo_id=SERVED, token=HF_TOKEN,
                  commit_message="promote v10 : poids B_ep6 + faisceau+LM "
                                 "(test 0,6177 vs greedy servi 0,6534 ; décision produit 26/08)")
print("PROMOTED →", SERVED, flush=True)

# Vérification post-push : l'arbre servi doit reproduire exactement celui tiré.
tree = {s.rfilename: s.size for s in api.model_info(SERVED, files_metadata=True).siblings}
for f in FILES:
    local = (out / f).stat().st_size
    assert tree.get(f) == local, f"{f} : servi={tree.get(f)} ≠ local={local}"
    print(f"vérifié : {f} ({local:,} o)", flush=True)
print("PROMOTION_VERIFIED", flush=True)
