# ══════════════════════════════════════════════════════════════════════════════
# KHIDMETI — ai-vision sur Modal (SigLIP2 zero-shot, tour image ONNX, CPU)
#
# Réutilise docker/ai-vision/server.py SANS modification : decode-at-size,
# protos raffinés + calibration Platt (cuits dans meta.json du repo),
# VISION_BIAS=0. Contrat inchangé :
#   POST /classify (octets image bruts, Content-Type: application/octet-stream)
#   → {profession, profession_confidence}
#
# Déploiement (depuis la racine du repo khid-back) :
#   modal deploy deploy/modal/vision_app.py
#
# Puis côté Render :
#   VISION_URL=https://<workspace>--khidmeti-vision-serve.modal.run
#
# Bascule future MobileCLIP2 (P5c, mesurée d'abord !) = VISION_REPO +
# VISION_MODEL_FILE dans STT_ENV-style dict ci-dessous — le repo SigLIP2
# servi reste intact.
# ══════════════════════════════════════════════════════════════════════════════

import subprocess

import modal

app = modal.App("khidmeti-vision")

# Mêmes dépendances que docker/ai-vision/Dockerfile — ne pas diverger.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "onnxruntime>=1.17,<2",
        "pillow>=10",
        "huggingface_hub>=0.23",
        "numpy>=1.24,<3",
    )
)

models = modal.Volume.from_name("khidmeti-models-vision", create_if_missing=True)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("khidmeti-hf")],
    volumes={"/models": models},
    mounts=[modal.Mount.local_dir(local_path="docker/ai-vision", remote_path="/app")],
    cpu=2,
    memory=2048,
    scaledown_window=300,
    timeout=900,
)
@modal.web_server(8015, startup_timeout=300)
def serve():
    # VISION_BIAS=0 = valeur prod depuis la bascule P4d (calibration Platt).
    # VISION_REPO par défaut du server.py = Walidrbh27/khidmeti-vision.
    subprocess.Popen(["python", "server.py"], cwd="/app")
