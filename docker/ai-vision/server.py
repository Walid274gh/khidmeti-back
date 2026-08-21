# docker/ai-vision/server.py
#
# Khidmeti vision — SigLIP2 zero-shot (P5 pipeline darija), tour image seule
# en ONNX int8 CPU + text_embeds précalculés (la tour texte est morte au
# moment de l'export Kaggle). Poids : repo HF privé Walidrbh27/khidmeti-vision,
# téléchargés au premier boot dans /models (volume ./docker/models/vision).
#
# VISION_REPO change de modèle sans toucher au code : tout (taille, mean/std,
# resize_mode, échelle/biais, fichier servi) est lu dans meta.json. P5c/P5d ont
# mesuré les candidats rapides (MC2 −1 pt + licence, patch32 −28 pts, TinyCLIP
# −7 pts) : aucun ne passe le gate ≤1 pt → SigLIP2 reste servi (août 2026).
# Bascule future = VISION_REPO + VISION_MODEL_FILE + VISION_BIAS, re-mesure
# paire obligatoire contre le servi, jamais de choix sur la fiche HF.
#
# API :
#   GET  /health                        → {"status":"ok"}
#   POST /classify  (image brute        → {"profession",
#        JPEG/PNG/WebP, ≤10 MB)            "profession_confidence","probs"}
#
# Score = sigmoïde SigLIP par classe (max sur les prompts de la classe) —
# probabilités indépendantes, calibrées par logit_scale/logit_bias du modèle.
# VISION_BIAS (float, défaut 0) : bouton de calibration terrain — si les
# photos correctes scorent sous la barre des 0.35 côté app, l'augmenter
# (+2/+3) relève toutes les sigmoïdes sans réexport.

import io
import json
import math
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
HF_REPO = os.environ.get("VISION_REPO", "Walidrbh27/khidmeti-vision")
# défaut : meta.json "serve_file" — le kernel d'export a mesuré si l'int8
# préserve les embeddings (cos>0.99) et a choisi ; la variable force au besoin
MODEL_FILE = os.environ.get("VISION_MODEL_FILE")
BIAS_KNOB = float(os.environ.get("VISION_BIAS", "0"))
PORT = int(os.environ.get("PORT", "8015"))
MAX_BYTES = 10 * 1024 * 1024  # même borne que le controller NestJS (multer)

FILES = ["model.int8.onnx", "model.fp32.onnx", "text_embeds.npy",
         "vision_labels.json", "meta.json"]


def ensure_weights():
    missing = [f for f in FILES if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if not missing:
        return
    print(f"[ai-vision] downloading {missing} from {HF_REPO} …", flush=True)
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=MODEL_DIR,
        allow_patterns=FILES,
        token=os.environ.get("HF_TOKEN") or None,
    )
    print("[ai-vision] weights ready", flush=True)


def load():
    import onnxruntime as ort

    with open(os.path.join(MODEL_DIR, "vision_labels.json"), encoding="utf-8") as f:
        labels = json.load(f)
    with open(os.path.join(MODEL_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    temb = np.load(os.path.join(MODEL_DIR, "text_embeds.npy"))  # [P, D] L2-norm
    model_file = MODEL_FILE or meta.get("serve_file", "model.int8.onnx")
    print(f"[ai-vision] serving {model_file}", flush=True)
    sess = ort.InferenceSession(
        os.path.join(MODEL_DIR, model_file),
        providers=["CPUExecutionProvider"],
    )
    return sess, temb, labels, meta


def preprocess(data, meta):
    # Réplique EXACTE du preprocessing gaté par assert dans le kernel Kaggle :
    # rescale 1/255, normalize mean/std, NCHW. resize_mode absent = "squash"
    # (SigLIP2 : resize direct en carré) ; "shortest" = petit côté puis crop
    # centré, la transformation native de MobileCLIP2 (P5c).
    h, w = meta["image_size"]
    resample = getattr(Image, meta["resample"].upper())
    img = Image.open(io.BytesIO(data))
    img.draft("RGB", (w * 4, h * 4))   # JPEG décodé à l'échelle (~1/4), pas le 12MP entier
    img = img.convert("RGB")
    if meta.get("resize_mode") == "shortest":
        short = min(img.size)
        nw, nh = max(w, int(img.width * w / short)), max(h, int(img.height * h / short))
        img = img.resize((nw, nh), resample)
        left, top = (nw - w) // 2, (nh - h) // 2
        img = img.crop((left, top, left + w, top + h))
    else:
        img = img.resize((w, h), resample)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.array(meta["image_mean"], dtype=np.float32)) \
        / np.array(meta["image_std"], dtype=np.float32)
    return x.transpose(2, 0, 1)[None]


def classify(data, sess, temb, labels, meta):
    emb = sess.run(None, {"pixel_values": preprocess(data, meta)})[0][0]  # [D]
    logits = meta["logit_scale"] * (temb @ emb) + meta["logit_bias"] + BIAS_KNOB
    classes = labels["classes"]
    per_cls = [-1e30] * len(classes)
    for logit, ci in zip(logits, labels["prompt_class"]):
        if logit > per_cls[ci]:
            per_cls[ci] = float(logit)
    probs = {c: round(1 / (1 + math.exp(-l)), 4) for c, l in zip(classes, per_cls)}
    best = max(range(len(classes)), key=lambda i: per_cls[i])
    return {
        "profession":            classes[best],
        "profession_confidence": probs[classes[best]],
        "probs":                 probs,
    }


def main():
    ensure_weights()
    sess, temb, labels, meta = load()

    # Self-check au boot : image de bruit → inférence complète, crash immédiat
    # si graphe/preprocessing/scoring est cassé (mieux qu'un 500 au premier
    # utilisateur).
    rng = np.random.default_rng(0)
    noise = Image.fromarray(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    buf = io.BytesIO()
    noise.save(buf, format="JPEG")
    smoke = classify(buf.getvalue(), sess, temb, labels, meta)
    assert smoke["profession"] in labels["classes"]
    print(f"[ai-vision] smoke: {smoke['profession']} "
          f"{smoke['profession_confidence']}", flush=True)

    if "--check" in sys.argv:  # usage: python server.py --check photo.jpg
        with open(sys.argv[-1], "rb") as f:
            print(json.dumps(classify(f.read(), sess, temb, labels, meta),
                             ensure_ascii=False))
        return

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/classify":
                return self._send(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length <= 0:
                    return self._send(400, {"error": "image body required"})
                if length > MAX_BYTES:
                    return self._send(413, {"error": "image too large"})
                self._send(200, classify(self.rfile.read(length),
                                         sess, temb, labels, meta))
            except Exception as e:  # jamais de crash serveur sur une requête
                self._send(500, {"error": str(e)[:200]})

        def log_message(self, fmt, *args):
            pass  # santé toutes les 20s — silence

    print(f"[ai-vision] listening on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
