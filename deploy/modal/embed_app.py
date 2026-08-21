# ══════════════════════════════════════════════════════════════════════════════
# KHIDMETI — ai-embed sur Modal (llama.cpp:server + nomic-embed-text-v1.5 Q8_0)
#
# Même binaire que le compose (image ghcr.io/ggml-org/llama.cpp:server), mêmes
# arguments que docker-compose.prod.yml (LLAMA_ARG_* traduits en flags) :
#   embeddings, pooling mean, ctx 2048, 2 threads, port 8012.
# Endpoint OpenAI-compatible : POST /v1/embeddings — contrat identique.
#
# Le modèle est PUBLIC (pas de HF_TOKEN requis). llama-server le télécharge
# au premier boot et le met en cache dans $XDG_CACHE_HOME → Volume persistant.
#
# Déploiement :
#   modal deploy deploy/modal/embed_app.py
#
# Puis côté Render (le service ajoute /embeddings, l'URL finit par /v1) :
#   EMBEDDINGS_URL=https://<workspace>--khidmeti-embed-serve.modal.run/v1
#   EMBEDDINGS_MODEL=nomic-embed-text-v1.5
# ══════════════════════════════════════════════════════════════════════════════

import os
import subprocess

import modal

app = modal.App("khidmeti-embed")

# Même image que docker-compose.prod.yml (ai-embed) + rien d'autre.
image = modal.Image.from_registry("ghcr.io/ggml-org/llama.cpp:server", add_python="3.11")

models = modal.Volume.from_name("khidmeti-models-embed", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/models": models},
    cpu=1,
    memory=1536,
    scaledown_window=300,
    timeout=900,
)
@modal.web_server(8012, startup_timeout=600)
def serve():
    env = {
        **os.environ,
        "HOME": "/models",             # cache llama.cpp → volume persistant
        "XDG_CACHE_HOME": "/models",
    }
    subprocess.Popen(
        [
            "llama-server",
            "--hf-repo",  "nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0",
            "--embeddings",
            "--pooling",  "mean",
            "--ctx-size", "2048",
            "--threads",  "2",
            "--host",     "0.0.0.0",
            "--port",     "8012",
        ],
        env=env,
    )
