# ══════════════════════════════════════════════════════════════════════════════
# KHIDMETI — ai-nlu sur Modal (dziriBERT 2 têtes, ONNX int8, CPU)
#
# Réutilise docker/ai-nlu/server.py SANS modification : même code, mêmes
# réponses JSON {intent, intent_confidence, profession, profession_confidence}.
# Les poids sont téléchargés au premier boot depuis le repo privé HF
# (HF_TOKEN via le Secret Modal « khidmeti-hf ») puis mis en cache dans un
# Volume persistant — les boots froids suivants ne re-téléchargent rien.
#
# Déploiement (depuis la racine du repo khid-back) :
#   modal deploy deploy/modal/nlu_app.py
#
# Après déploiement, copier l'URL affichée dans NLU_URL côté Render :
#   NLU_URL=https://<workspace>--khidmeti-nlu-serve.modal.run
# (l'API NestJS ajoute elle-même /classify et /health — aucun changement code)
# ══════════════════════════════════════════════════════════════════════════════

import subprocess

import modal

app = modal.App("khidmeti-nlu")

# Mêmes dépendances que docker/ai-nlu/Dockerfile — ne pas diverger.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "onnxruntime>=1.17,<2",
        "tokenizers>=0.15",
        "huggingface_hub>=0.23",
        "numpy>=1.24,<3",
    )
)

models = modal.Volume.from_name("khidmeti-models-nlu", create_if_missing=True)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("khidmeti-hf")],  # contient HF_TOKEN
    volumes={"/models": models},                      # MODEL_DIR par défaut
    mounts=[modal.Mount.local_dir(local_path="docker/ai-nlu", remote_path="/app")],
    cpu=1,
    memory=1024,
    scaledown_window=300,   # reste chaud 5 min après le dernier appel
    timeout=600,
)
@modal.web_server(8013, startup_timeout=300)
def serve():
    subprocess.Popen(["python", "server.py"], cwd="/app")
