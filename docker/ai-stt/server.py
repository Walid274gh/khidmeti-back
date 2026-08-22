# docker/ai-stt/server.py
#
# Khidmeti STT — faster-whisper int8, CPU (P4 pipeline darija).
# Modèle via STT_MODEL : taille builtin ("small"/"medium") OU repo HF CTranslate2
# (ex. Walidrbh27/khidmeti-stt, fine-tune Casablanca-Algeria P4c — privé, lu via
# HF_TOKEN env). Poids téléchargés au premier boot dans /models (volume
# ./docker/models/stt — jamais dans l'image ni GitHub).
#
# v6 (Aug 10 2026) MISE AU RANCART (Aug 15 2026) : STT_MODEL avait été mis à
# anaszil/whisper-large-v3-turbo-darija « pré-entraîné darija, WER <0,40 attendu ».
# Les deux affirmations étaient fausses : ce dépôt est un adaptateur PEFT que
# faster-whisper ne sait pas charger (→ ai-stt en FALLBACK silencieux), et le
# 0,40 n'a jamais été mesuré. Mesuré depuis, sur 831 clips Casablanca-Algeria :
# v3 servi 0,6300 · openai/whisper-large-v3 nu 0,8055. Défaut = nos poids.
#
# v8 (Aug 17 2026) : STT_ENGINE choisit le moteur, même API, même JSON.
#   whisper (défaut) → faster-whisper, ce qui tourne aujourd'hui (WER 0,6097
#                      mesuré, réglages prod, run khidmeti-stt-paired)
#   ctc              → wav2vec2-XLSR darija exporté en ONNX par
#                      ml/kaggle/stt_kernel_v8_ctc.py : une seule passe encodeur,
#                      coût ∝ durée réelle au lieu des 30 s de padding de whisper.
# v10-ctc-lm (Aug 20 2026) — DÉCISION PRODUIT : Khidmeti a besoin de latence et
# le NLU lit le sens, pas la graphie → STT_ENGINE=ctc même si le gate WER (0,6097)
# n'est pas battu. Le décodage MAXÉ est faisceau+LM (ctc_decode.py, portage exact
# du kernel) : greedy 0,6688 → +LM 0,6236 CER 0,2310, 0,231×RT = ×26 vs whisper.
# Repo = khidmeti-stt-ctc (stable ; restauré depuis l'état connu-bon 63c09cef —
# poids B_ep8 24 h + LM ordre 3 + domaine 214 + vocab 193). La config décodage
# (order, mix, α, β, floor, beam) est lue de meta.json à chaque boot : un run
# futur peut servir un LM ordre-4 mélangé SANS toucher au code.
# Retour arrière = retirer STT_ENGINE : le repo whisper reste intact.
#
# API :
#   GET  /health                       → {"status":"ok"}
#   POST /transcribe  (audio brut)     → {"text","language",
#        Content-Type: audio/*            "language_probability","duration"}
#
# PyAV (dépendance de faster-whisper) décode m4a/ogg/wav/mp3/webm nativement —
# aucun transcodage côté API. Langue auto-détectée (darija → "ar", français
# → "fr") ; STT_LANGUAGE force une langue si l'auto-détection déçoit sur le
# terrain. v6 uses pre-trained darija model = much better baseline than vanilla whisper.

import io
import json
import math
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np                    # dépendance de faster-whisper ET d'onnxruntime

MODEL_DIR  = os.environ.get("MODEL_DIR", "/models")
ENGINE     = os.environ.get("STT_ENGINE", "whisper").lower()   # "ctc" = wav2vec2 ONNX
_DEFAULT_MODEL = ("Walidrbh27/khidmeti-stt-ctc" if ENGINE == "ctc"
                  else "Walidrbh27/khidmeti-stt")
MODEL_SIZE = os.environ.get("STT_MODEL") or _DEFAULT_MODEL  # "" (compose) → défaut
# Piège : .env fixe STT_MODEL=…/khidmeti-stt (whisper). Basculer STT_ENGINE seul
# ferait chercher meta.json dans le repo whisper → 404 illisible au démarrage.
if ENGINE == "ctc" and MODEL_SIZE.endswith("/khidmeti-stt"):
    sys.exit("[ai-stt] STT_ENGINE=ctc mais STT_MODEL pointe le repo whisper "
             f"({MODEL_SIZE}) : mettre STT_MODEL=Walidrbh27/khidmeti-stt-ctc "
             "ou retirer STT_MODEL.")
MODEL_FILE = os.environ.get("STT_MODEL_FILE")  # ctc : None = serve_file de meta.json
LANGUAGE   = os.environ.get("STT_LANGUAGE") or None   # None = auto-détection
# Décodage CTC faisceau+LM. SOURCE DE VÉRITÉ = meta.json du repo (écrit par le
# kernel à chaque run : order, is_mix, w_ecrit, alpha, beta, floor, beam, prune).
# Les valeurs MESURÉES du run khidmeti-stt-lm-v10 (24h, LM écrit seul, domaine) :
# order=3 alpha=0.3 beta=2.0 floor=-30 beam=128 → 0,6236. Les env STT_LM_* ne
# servent QU'À SURRIDER le meta en cas de besoin — jamais de valeur fixe ici,
# sinon un futur run (ex. LM order-4 mélangé) casse le serveur au changement
# de repo : exactement le bug du 20/08 (assert order==3 + mix non chargé).
LM_FILE      = os.environ.get("STT_LM_FILE") or "lm.npz"
LM_PARLE     = os.environ.get("STT_LM_PARLE") or "lm_parle.npz"  # mix : 2e fichier
LM_ALPHA     = float(os.environ["STT_LM_ALPHA"]) if os.environ.get("STT_LM_ALPHA") else None
LM_BETA      = float(os.environ["STT_LM_BETA"])  if os.environ.get("STT_LM_BETA")  else None
LM_FLOOR     = float(os.environ["STT_LM_FLOOR"]) if os.environ.get("STT_LM_FLOOR") else None
LM_BEAM      = int(os.environ["STT_LM_BEAM"])    if os.environ.get("STT_LM_BEAM")    else None
THREADS      = int(os.environ.get("STT_THREADS", "4"))
PORT       = int(os.environ.get("PORT", "8014"))
MAX_BYTES  = 16 * 1024 * 1024
BEAM       = int(os.environ.get("STT_BEAM", "1"))     # 5 = +qualité, ~3× CPU
# ponytail: ctc = attention O(n²) sur tout le clip → on tronque ; chunker si on
# transcrit un jour du long format (les requêtes vocales font 3–10 s). 30 s tient
# dans le mem_limit 2g de prod (≈144 Mo d'attention) ; 60 s en demanderait ~576.
MAX_SECONDS = float(os.environ.get("STT_MAX_SECONDS", "30"))
# Amorce de vocabulaire métier : whisper la traite comme contexte précédent et
# biaise le décodage vers ces mots (fix constaté : «خاصني بلومبي» transcrit en
# charabia sans amorce). Vide/absent = défaut ci-dessous ; STT_PROMPT=off désactive.
_DEFAULT_PROMPT = (
    "خاصني بلومبي كهربائي سباك نجار حداد سودور بناي ماصون صباغ حلاق كوافور "
    "خياط طباخة جباص بلاكو ميكانيسيان كليماتيزور فريجيدار ماشينة تنظيف "
    "ديمناجمون تصليح نحوس على واحد يجيني للدار"
)
PROMPT = os.environ.get("STT_PROMPT") or _DEFAULT_PROMPT
if PROMPT.strip().lower() == "off":
    PROMPT = None

# ponytail: verrou global — une transcription à la fois, borne la RAM sur la
# machine 8 GB ; passer à un pool si le trafic voix le justifie un jour.
_lock = threading.Lock()


def load():
    from faster_whisper import WhisperModel
    print(f"[ai-stt] loading whisper-{MODEL_SIZE} int8 "
          f"(premier boot = téléchargement des poids) …", flush=True)
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8",
                         cpu_threads=THREADS, download_root=MODEL_DIR)
    print("[ai-stt] model ready", flush=True)
    return model


def transcribe(model, data):
    with _lock:
        segments, info = model.transcribe(
            io.BytesIO(data),
            language=LANGUAGE,
            beam_size=BEAM,
            vad_filter=True,    # silero VAD — coupe les silences, limite les hallucinations
            initial_prompt=PROMPT,
            condition_on_previous_text=False,  # clips courts : un segment garblé ne contamine pas le suivant
        )
        text = " ".join(s.text.strip() for s in segments).strip()
    return {
        "text":                 text,
        "language":             info.language,
        "language_probability": round(float(info.language_probability), 4),
        "duration":             round(float(info.duration), 2),
    }


# ══ moteur ctc (wav2vec2-XLSR darija, ONNX) ═══════════════════════════════════
# Le décodage et la normalisation sont REJOUÉS TELS QUELS depuis
# ml/kaggle/stt_kernel_v8_ctc.py (asserts de parité côté kernel) : toute
# divergence ici casse silencieusement le WER mesuré. Le faisceau+LM vient de
# ctc_decode.py, copie identique de stt_lm_kernel_v9.py (démo 7 tests au boot).

import ctc_decode


def decode_logits(logits, id2tok, blank, special, lm, cfg):
    """logits CTC (n_frames × vocab) → texte faisceau+LM. cfg = paramètres
    décodage RÉSOLUS (meta du modèle, env en override) — toujours passés,
    jamais les globals : un LM order-4 mélangé et un order-3 écrit n'ont pas
    les mêmes α/β/beam optimaux."""
    rows = ctc_decode.log_softmax(logits.tolist() if hasattr(logits, "tolist")
                                  else logits)
    return ctc_decode.beam_decode(
        rows, id2tok, blank, special, lm=lm,
        alpha=cfg["alpha"], beta=cfg["beta"], floor=cfg["floor"],
        beam=cfg["beam"], delim="|")             # delim du vocab, asserté au load


def _resolve_lm_cfg(meta_lm):
    """Paramètres décodage : meta (source de vérité) ← env (override) ← défauts."""
    def _pick(env, meta_key, default):
        if env is not None:
            return env
        return meta_lm.get(meta_key, default)
    return {
        "alpha": _pick(LM_ALPHA, "alpha", 0.3),
        "beta":  _pick(LM_BETA, "beta", 2.0),
        "floor": _pick(LM_FLOOR, "floor", -30.0),
        "beam":  _pick(LM_BEAM, "beam", 128),
        "prune": float(meta_lm.get("prune", 1e-4)),
    }


def _ctc_lm_load(pull, meta):
    """LM à SERVIR depuis les .npz du repo, config VENANT DU META (jamais en
    dur — le bug 20/08 : le run 93h a servi un LM order-4 mélangé et l'ancien
    serveur "assert order==3" plantait au boot → FALLBACK → 0 %)."""
    ml = meta.get("lm") or {}
    is_mix = bool(ml.get("is_mix"))
    lm = ctc_decode.load_lm_cfg(pull(LM_FILE),
                                pull(LM_PARLE) if is_mix else None, ml)
    cfg = _resolve_lm_cfg(ml)
    return lm, cfg, is_mix


def load_ctc():
    # meta.json d'abord : il dit quel fichier servir. Un snapshot_download de la
    # racine tirerait aussi le fp32 mort (~1,3 Go) — on prend 3 fichiers, pas plus.
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    def pull(name):
        return hf_hub_download(MODEL_SIZE, name, local_dir=MODEL_DIR,
                               token=os.environ.get("HF_TOKEN") or None)

    with open(pull("meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(pull("ctc_vocab.json"), encoding="utf-8") as f:
        v = json.load(f)
    fname = MODEL_FILE or meta.get("serve_file", "model.int8.onnx")
    print(f"[ai-stt] ctc {meta.get('version', '?')} : {fname}", flush=True)

    opt = ort.SessionOptions()
    opt.intra_op_num_threads = THREADS
    # Render free = 512 Mi hard limit : l'arène d'allocation + l'optimisation
    # graphique au chargement doublent le pic mémoire (~2x la taille du modèle).
    opt.enable_cpu_mem_arena = False
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess = ort.InferenceSession(pull(fname), opt, providers=["CPUExecutionProvider"])
    id2tok, blank, special = v["id2tok"], v["blank_id"], set(v.get("special", []))
    assert v.get("delim") == "|", v.get("delim")

    lm, cfg, is_mix = _ctc_lm_load(pull, meta)
    ngrams = (len(lm["mix"][0]["keys"]) + len(lm["mix"][1]["keys"]) if is_mix
              else len(lm["keys"]))
    print(f"[ai-stt] LM {lm['order']}-gram {'mélangé' if is_mix else 'écrit seul'} "
          f"{ngrams} ngrams (α={cfg['alpha']} β={cfg['beta']} floor={cfg['floor']} "
          f"beam={cfg['beam']})", flush=True)
    return sess, id2tok, blank, special, lm, cfg


def transcribe_ctc(state, data):
    from faster_whisper.audio import decode_audio     # PyAV : m4a/ogg/webm/wav/mp3

    sess, id2tok, blank, special, lm, cfg = state
    x = np.asarray(decode_audio(io.BytesIO(data), sampling_rate=16000),
                   dtype=np.float32)
    dur = len(x) / 16000.0
    x = x[:int(MAX_SECONDS * 16000)]
    if len(x) < 400:                                  # < 25 ms : rien à décoder
        return {"text": "", "language": LANGUAGE or "ar",
                "language_probability": 1.0, "duration": round(dur, 2)}
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)      # do_normalize Wav2Vec2
    with _lock:
        logits = sess.run(None, {"input_values": x[None]})[0]
    return {
        "text":                 decode_logits(logits[0], id2tok, blank, special, lm, cfg),
        "language":             LANGUAGE or "ar",     # modèle mono-langue
        "language_probability": 1.0,
        "duration":             round(dur, 2),
    }


def _silence_wav(seconds=1):
    # 16 kHz mono 16-bit de silence — smoke test sans fichier externe.
    import wave
    buf = io.BytesIO()
    w = wave.open(buf, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    w.writeframes(b"\x00\x00" * 16000 * seconds)
    w.close()
    return buf.getvalue()


def main():
    # Parité du portage d'abord (crash immédiat si ctc_decode diverge du kernel)
    if ENGINE == "ctc":
        ctc_decode.demo()

    state, run = (load_ctc(), transcribe_ctc) if ENGINE == "ctc" else (load(), transcribe)

    # Self-check au boot : décode + transcrit 1s de silence, crash immédiat si
    # PyAV/CTranslate2/VAD (ou onnxruntime/vocab côté ctc) est cassé — mieux
    # qu'un 500 au premier utilisateur.
    smoke = run(state, _silence_wav())
    assert isinstance(smoke["text"], str)
    print(f"[ai-stt] smoke: {smoke}", flush=True)

    if "--check" in sys.argv:  # usage: python server.py --check fichier.wav
        with open(sys.argv[-1], "rb") as f:
            print(json.dumps(run(state, f.read()), ensure_ascii=False))
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
            if self.path != "/transcribe":
                return self._send(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length <= 0:
                    return self._send(400, {"error": "audio body required"})
                if length > MAX_BYTES:
                    return self._send(413, {"error": "audio too large"})
                self._send(200, run(state, self.rfile.read(length)))
            except Exception as e:  # jamais de crash serveur sur une requête
                self._send(500, {"error": str(e)[:200]})

        def log_message(self, fmt, *args):
            pass  # santé toutes les 20s — silence

    print(f"[ai-stt] listening on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
