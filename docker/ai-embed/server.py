import os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = None

@app.on_event("startup")
def load():
    global model
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    print("embed ready", flush=True)

class Req(BaseModel):
    model: str = ""
    input: str | list[str] = ""

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/v1/models")
def models():
    return {"data": [{"id": "nomic-embed-text-v1.5"}]}

@app.post("/v1/embeddings")
def embed(req: Req):
    texts = req.input if isinstance(req.input, list) else [req.input]
    vecs = model.encode(texts, normalize_embeddings=True).tolist()
    return {"data": [{"embedding": v, "index": i} for i, v in enumerate(vecs)], "model": req.model or "nomic-embed-text-v1.5"}

@app.post("/embeddings")
def embed2(req: Req):
    return embed(req)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8012)))
