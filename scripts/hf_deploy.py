#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
# scripts/hf_deploy.py — deploy backend to HuggingFace Docker Spaces (free MVP
# hosting, replaces ngrok). Stdlib only — runs anywhere python3 exists.
#
#   python3 scripts/hf_deploy.py deploy    # create Spaces + secrets + push code
#   python3 scripts/hf_deploy.py status    # one-shot runtime stage + /health
#
# Topology (mirrors docker-compose.prod.yml minus nginx):
#   Walidrbh27/khidmeti-api       ← apps/api verbatim (public repo — NO secrets
#                                    in source; env arrives via Space secrets)
#   Walidrbh27/khidmeti-ai-embed  ← llama.cpp:server + nomic-embed (API-key gated)
#
# Secrets come from ../.env. EMBEDDINGS_API_KEY is generated on first run and
# persisted back into .env so api↔embed stay in sync across re-runs.
# Re-running `deploy` is safe: create=409-tolerated, secrets upsert, push=new commit.
# ══════════════════════════════════════════════════════════════════════════════
import base64
import json
import secrets as pysecrets
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API_DIR = ROOT / 'apps' / 'api'
HUB = 'https://huggingface.co'
NS = 'Walidrbh27'
API_SPACE = 'khidmeti-api'
EMBED_SPACE = 'khidmeti-ai-embed'
API_URL = f'https://{NS.lower()}-{API_SPACE}.hf.space'
EMBED_URL = f'https://{NS.lower()}-{EMBED_SPACE}.hf.space'

API_SECRETS = [
    'MONGODB_URI', 'FIELD_ENC_KEY', 'FIELD_ENC_PEPPER',
    'CLOUDINARY_CLOUD_NAME', 'CLOUDINARY_API_KEY', 'CLOUDINARY_API_SECRET',
    'FIREBASE_PROJECT_ID', 'FIREBASE_CLIENT_EMAIL', 'FIREBASE_PRIVATE_KEY',
    'QDRANT_URL', 'QDRANT_API_KEY', 'REDIS_URL', 'EMBEDDINGS_API_KEY',
]
API_VARIABLES = {
    'CORS_ORIGINS': 'https://khidmeti.pages.dev,http://localhost:3000,http://localhost:8080',
    'EMBEDDINGS_URL': f'{EMBED_URL}/v1',
    'EMBEDDINGS_MODEL': 'nomic-embed-text-v1.5',
}

API_README = f"""---
title: Khidmeti API
emoji: 🛠️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 3000
pinned: false
---

# Khidmeti API

NestJS backend for the Khidmeti home-services platform (Algeria).
State lives in MongoDB Atlas / Cloudinary / Qdrant Cloud / Firebase —
this Space is stateless and safe to restart.
"""

EMBED_README = """---
title: Khidmeti AI Embed
emoji: 🧲
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Khidmeti AI Embed

llama.cpp embeddings server (nomic-embed-text-v1.5 Q8_0), API-key gated.
"""

# HF runs containers as uid 1000 with no writable $HOME guarantee → cache in /tmp.
EMBED_DOCKERFILE = """FROM ghcr.io/ggml-org/llama.cpp:server
ENV HOME=/tmp \\
    LLAMA_CACHE=/tmp/llama-cache \\
    LLAMA_ARG_HF_REPO=nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0 \\
    LLAMA_ARG_EMBEDDINGS=true \\
    LLAMA_ARG_POOLING=mean \\
    LLAMA_ARG_CTX_SIZE=2048 \\
    LLAMA_ARG_PORT=7860 \\
    LLAMA_ARG_HOST=0.0.0.0 \\
    LLAMA_ARG_THREADS=2
"""


def load_env() -> dict:
    env = {}
    for line in (ROOT / '.env').read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, _, v = line.partition('=')
            env[k.strip()] = v.strip()
    return env


def ensure_embed_key(env: dict) -> dict:
    if not env.get('EMBEDDINGS_API_KEY'):
        key = pysecrets.token_hex(32)
        path = ROOT / '.env'
        text = path.read_text()
        if 'EMBEDDINGS_API_KEY=' in text:
            text = text.replace('EMBEDDINGS_API_KEY=', f'EMBEDDINGS_API_KEY={key}', 1)
        else:
            text += f'\nEMBEDDINGS_API_KEY={key}\n'
        path.write_text(text)
        env['EMBEDDINGS_API_KEY'] = key
        print('• generated EMBEDDINGS_API_KEY and persisted to .env')
    return env


def hf(env: dict, method: str, path: str, body=None, ndjson: str | None = None):
    headers = {'Authorization': f"Bearer {env['HF_TOKEN']}"}
    data = None
    if ndjson is not None:
        headers['Content-Type'] = 'application/x-ndjson'
        data = ndjson.encode()
    elif body is not None:
        headers['Content-Type'] = 'application/json'
        data = json.dumps(body).encode()
    req = urllib.request.Request(f'{HUB}{path}', data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as res:
            return res.status, json.loads(res.read() or b'{}')
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b'{}')


def create_space(env: dict, name: str):
    status, res = hf(env, 'POST', '/api/repos/create',
                    {'type': 'space', 'name': name, 'private': False, 'sdk': 'docker'})
    if status == 200:
        print(f'• created Space {NS}/{name}')
    elif status == 409:
        print(f'• Space {NS}/{name} already exists')
    else:
        sys.exit(f'FATAL create {name}: {status} {res}')


def put_kv(env: dict, space: str, kind: str, key: str, value: str):
    status, res = hf(env, 'POST', f'/api/spaces/{NS}/{space}/{kind}', {'key': key, 'value': value})
    if status not in (200, 201):
        sys.exit(f'FATAL {kind[:-1]} {key} on {space}: {status} {res}')


def commit(env: dict, space: str, files: dict[str, bytes], summary: str):
    lines = [json.dumps({'key': 'header', 'value': {'summary': summary, 'description': ''}})]
    for path, content in sorted(files.items()):
        lines.append(json.dumps({'key': 'file', 'value': {
            'path': path, 'encoding': 'base64',
            'content': base64.b64encode(content).decode(),
        }}))
    status, res = hf(env, 'POST', f'/api/spaces/{NS}/{space}/commit/main', ndjson='\n'.join(lines))
    if status != 200:
        sys.exit(f'FATAL commit {space}: {status} {res}')
    print(f'• pushed {len(files)} files to {NS}/{space}')


def api_files() -> dict[str, bytes]:
    files = {'README.md': API_README.encode()}
    for name in ('Dockerfile', '.dockerignore', 'package.json', 'nest-cli.json', 'tsconfig.json'):
        files[name] = (API_DIR / name).read_bytes()
    for p in sorted((API_DIR / 'src').rglob('*')):
        rel = p.relative_to(API_DIR)
        # mirror .dockerignore: no dev seed scripts, no tests in the public repo
        if p.is_dir() or 'scripts' in rel.parts or p.name.endswith(('.spec.ts', '.e2e-spec.ts')):
            continue
        files[str(rel)] = p.read_bytes()
    return files


def runtime(env: dict, space: str) -> str:
    _, res = hf(env, 'GET', f'/api/spaces/{NS}/{space}/runtime')
    return res.get('stage', 'UNKNOWN')


def cmd_deploy():
    env = ensure_embed_key(load_env())
    for name in (EMBED_SPACE, API_SPACE):
        create_space(env, name)
    put_kv(env, EMBED_SPACE, 'secrets', 'LLAMA_ARG_API_KEY', env['EMBEDDINGS_API_KEY'])
    for key in API_SECRETS:
        if env.get(key):
            put_kv(env, API_SPACE, 'secrets', key, env[key])
    for key, value in API_VARIABLES.items():
        put_kv(env, API_SPACE, 'variables', key, value)
    print('• secrets/variables set')
    commit(env, EMBED_SPACE, {'README.md': EMBED_README.encode(),
                              'Dockerfile': EMBED_DOCKERFILE.encode()}, 'deploy ai-embed')
    commit(env, API_SPACE, api_files(), 'deploy api')
    print(f'done — building now:\n  {HUB}/spaces/{NS}/{API_SPACE}\n  {HUB}/spaces/{NS}/{EMBED_SPACE}')
    print(f'api will serve at {API_URL} (health: {API_URL}/health)')


def cmd_status():
    env = load_env()
    for name in (API_SPACE, EMBED_SPACE):
        print(f'{name}: {runtime(env, name)}')
    for url in (f'{API_URL}/health', f'{EMBED_URL}/health'):
        try:
            with urllib.request.urlopen(url, timeout=15) as res:
                print(f'{url} -> {res.status}')
        except Exception as e:  # noqa: BLE001 — status probe, any failure is the answer
            print(f'{url} -> {type(e).__name__}: {e}')


if __name__ == '__main__':
    {'deploy': cmd_deploy, 'status': cmd_status}.get(
        sys.argv[1] if len(sys.argv) > 1 else '',
        lambda: sys.exit('usage: hf_deploy.py deploy|status'))()
