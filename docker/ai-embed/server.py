import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json, urllib.request

# Proxy to HF Inference API — zero build, zero model download
HF_TOKEN = os.environ.get("HF_TOKEN", "")
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PORT = int(os.environ.get("PORT", "8012"))

def hf_embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    # Use HF Inference API
    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBED_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    data = json.dumps({"inputs": texts, "options": {"wait_for_model": True}}).encode()
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        vecs = json.loads(r.read())
    # HF returns list of lists for batch
    if vecs and isinstance(vecs[0], (int, float)):
        vecs = [vecs]
    return vecs

class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        if self.path in ("/health", "/v1/models", "/models"):
            self._send(200, {"status": "ok", "data": [{"id": "nomic-embed-text-v1.5"}]})
        else:
            self._send(404, {"error": "not found"})
    def do_POST(self):
        if self.path not in ("/v1/embeddings", "/embeddings", "/v1/embed"):
            return self._send(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            inp = body.get("input", body.get("inputs", ""))
            model = body.get("model", "nomic-embed-text-v1.5")
            texts = inp if isinstance(inp, list) else [inp]
            vecs = hf_embed(texts)
            self._send(200, {"data": [{"embedding": v, "index": i} for i, v in enumerate(vecs)], "model": model, "object": "list"})
        except Exception as e:
            self._send(500, {"error": str(e)[:300]})
    def log_message(self, fmt, *args): pass

print(f"[embed] proxy -> {EMBED_MODEL} on :{PORT}", flush=True)
ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
