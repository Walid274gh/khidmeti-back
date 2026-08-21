# docker/ai-nlu/server.py
#
# Khidmeti NLU — dziriBERT 2 têtes (intent + profession), ONNX int8, CPU.
# Poids : repo HF privé Walidrbh27/khidmeti-nlu, téléchargés au premier boot
# dans /models (volume ./docker/models/nlu — jamais dans l'image ni GitHub).
#
# API :
#   GET  /health            → {"status":"ok"}
#   POST /classify {"text"} → {"intent","intent_confidence",
#                              "profession","profession_confidence"}
#
# Le tokenizer réplique EXACTEMENT les conditions d'éval du kernel Kaggle :
# truncation + padding à max_len (64) — le modèle int8 a été gaté ainsi.

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
HF_REPO   = os.environ.get("NLU_REPO", "Walidrbh27/khidmeti-nlu")
PORT      = int(os.environ.get("PORT", "8013"))

FILES = ["model.int8.onnx", "tokenizer.json", "tokenizer_config.json",
         "labels.json", "meta.json"]


def ensure_weights():
    missing = [f for f in FILES if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if not missing:
        return
    print(f"[ai-nlu] downloading {missing} from {HF_REPO} …", flush=True)
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=MODEL_DIR,
        allow_patterns=FILES,
        token=os.environ.get("HF_TOKEN") or None,
    )
    print("[ai-nlu] weights ready", flush=True)


def load():
    import onnxruntime as ort
    from tokenizers import Tokenizer

    with open(os.path.join(MODEL_DIR, "labels.json"), encoding="utf-8") as f:
        labels = json.load(f)
    with open(os.path.join(MODEL_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    max_len = int(meta.get("max_len", 64))

    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    pad_id = tok.token_to_id("[PAD]") or 0
    tok.enable_truncation(max_length=max_len)
    tok.enable_padding(length=max_len, pad_id=pad_id, pad_token="[PAD]")

    sess = ort.InferenceSession(
        os.path.join(MODEL_DIR, "model.int8.onnx"),
        providers=["CPUExecutionProvider"],
    )
    return tok, sess, labels


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def classify(text, tok, sess, labels):
    enc  = tok.encode(text)
    ids  = np.array([enc.ids], dtype=np.int64)
    mask = np.array([enc.attention_mask], dtype=np.int64)
    intent_logits, prof_logits = sess.run(
        None, {"input_ids": ids, "attention_mask": mask})
    pi = softmax(intent_logits[0])
    pp = softmax(prof_logits[0])
    return {
        "intent":                labels["intents"][int(pi.argmax())],
        "intent_confidence":     round(float(pi.max()), 4),
        "profession":            labels["professions"][int(pp.argmax())],
        "profession_confidence": round(float(pp.max()), 4),
    }


def main():
    ensure_weights()
    tok, sess, labels = load()

    # Self-check au boot : une inférence de fumée, crash immédiat si le graphe
    # ou le tokenizer est cassé (mieux qu'un 500 au premier utilisateur).
    smoke = classify("راني نحوس على بلومبي", tok, sess, labels)
    assert smoke["profession"] in labels["professions"]
    print(f"[ai-nlu] smoke: {smoke}", flush=True)

    if "--check" in sys.argv:  # usage: python server.py --check "texte"
        print(json.dumps(classify(sys.argv[-1], tok, sess, labels),
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
                data = json.loads(self.rfile.read(length) or b"{}")
                text = str(data.get("text", "")).strip()[:2000]
                if not text:
                    return self._send(400, {"error": "text required"})
                self._send(200, classify(text, tok, sess, labels))
            except Exception as e:  # jamais de crash serveur sur une requête
                self._send(500, {"error": str(e)[:200]})

        def log_message(self, fmt, *args):
            pass  # santé toutes les 20s — silence

    print(f"[ai-nlu] listening on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
