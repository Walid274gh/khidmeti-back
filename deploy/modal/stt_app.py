# ══════════════════════════════════════════════════════════════════════════════
# KHIDMETI — ai-stt sur Modal (wav2vec2-CTC ONNX int8 + beam/LM stdlib)
#
# Réutilise docker/ai-stt/server.py + ctc_decode.py SANS modification.
# Contrat inchangé : POST /transcribe (octets bruts, Content-Type = mime audio)
# → {text, language, duration}.
#
# ⚠️ SOURCE DE VÉRITÉ des paramètres LM = meta.json du repo HF (écrit par le
# kernel à chaque run). Les STT_LM_* restent VIDES exprès — une valeur en dur
# ici écraserait le meta (bug du 20/08, LM ordre-4 mélangé).
#
# Déploiement (depuis la racine du repo khid-back) :
#   modal deploy deploy/modal/stt_app.py
#
# Puis côté Render :
#   STT_URL=https://<workspace>--khidmeti-stt-serve.modal.run
# ══════════════════════════════════════════════════════════════════════════════

import os
import subprocess

import modal

app = modal.App("khidmeti-stt")

# Mêmes dépendances que docker/ai-stt/Dockerfile (PyAV arrive avec
# faster-whisper ; kenlm absent = fallback stdlib de ctc_decode.py, voulu).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "faster-whisper>=1.0,<2",
        "onnxruntime>=1.17,<2",
    )
)

models = modal.Volume.from_name("khidmeti-models-stt", create_if_missing=True)

# Bascule v8 APPLIQUÉE le 2026-08-20 (décision produit : latence > 1,4 pt WER).
# Retour arrière whisper : STT_ENGINE="whisper" + STT_MODEL repo khidmeti-stt.
STT_ENV = {
    "STT_ENGINE":      "ctc",
    "STT_MODEL":       "Walidrbh27/khidmeti-stt-ctc",
    "STT_MODEL_FILE":  "model.int8.onnx",
    "STT_LANGUAGE":    "ar",
    "STT_MAX_SECONDS": "30",
    # Beam+LM : meta.json décide — ne rien fixer ici.
    "STT_LM_FILE": "", "STT_LM_ALPHA": "", "STT_LM_BETA": "",
    "STT_LM_FLOOR": "", "STT_LM_BEAM": "",
}


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("khidmeti-hf")],
    volumes={"/models": models},
    mounts=[modal.Mount.local_dir(local_path="docker/ai-stt", remote_path="/app")],
    cpu=2,
    memory=2560,
    scaledown_window=300,
    timeout=900,
)
@modal.web_server(8014, startup_timeout=600)  # chargement modèle+LM ≈ start-period 600s
def serve():
    env = {**os.environ, **STT_ENV}
    subprocess.Popen(["python", "server.py"], cwd="/app", env=env)
