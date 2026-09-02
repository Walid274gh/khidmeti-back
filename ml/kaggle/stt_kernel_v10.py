#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
# ml/kaggle/stt_kernel_v10.py — v10 : on ne change PAS de famille, on donne au
# modèle la DONNÉE qui lui manquait.
#
# Constat (mesuré, MÊMES 831 clips Casablanca-Algeria, MÊME moteur int8 4 threads) :
#   whisper medium+LoRA servi ....... WER 0,6300  CER 0,2373
#   CTC v8 greedy ................... WER 0,6914  CER 0,2531   ← v8 (749 énoncés
#     DZ uniques, 1,5 h d'audio) : val PLAT + loss qui baisse = surapprentissage.
#     Le plafond de v8 était la DONNÉE, pas le calcul (6,8 h GPU non consommées).
#   CTC v8 + LM 4-gram (v9a) ........ WER 0,6505  CER 0,2444   ← le LM répare le
#     décodage mais plafonne aussi (corpus ÉCRIT, registre ≠ parlé) ; il ne peut
#     pas toucher au CER. Les gains restants sont dans le MODÈLE + la DONNÉE.
#
# v10 = RÉENTRAÎNEMENT sur ~56 k clips algériens VÉRIFIÉS (marqueurs dialecte,
# session v9a), cuits par stt_prep_v10.py (kernel CPU) en un dataset Kaggle
# privé khidmeti-stt-v10 : dédoublonnage, garde-fou anti-fuite vs le test,
# parts sources total 24 h d'audio → flac 16 k + metadata.csv. Ici on réutilise la machine
# d'entraînement v8 (2 étages, vocabulaire vérifié, ONNX, gate) À L'IDENTIQUE,
# seule l'alimentation de l'étage B change :
#   v8 : Casablanca:Algeria ×2 (1 492 items ≈ 2 h d'audio)
#   v11 : Casablanca:Algeria ×2 + ~11 k clips v10 (24 h d'audio, 16× la donnée v8)
#
# Discpline de comparabilité : SEED 20260817 et ORDRE des tirages IDENTIQUES à
# v8 — la découpe val/test (83/831) reste bit-identique ; le loader v10 ne
# consomme AUCUN RNG global (lecture CSV locale, items triés par ordre du
# fichier), le shuffle d'époque se fait dans stage() comme avant.
#
# GATE = WER(candidat, greedy int8, mêmes 831 clips) < 0,6300.
#   Le décodage de service est greedy (docker/ai-stt/server.py) : le gate reste
#   greedy. Le LM 4-gram (stt_lm_kernel_v9.py) se REJOUERA tel quel sur les
#   poids v10 si le greedy passe proche : poids indépendants du décodage.
#
# Push (GPU) — AUCUN `DATASET=` : le v10 est tiré de HF par snapshot_download
# (ligne ~274), pas monté en dataset Kaggle. `DATASET=Walidrbh27/…` irait dans
# datasetDataSources, où Kaggle attend un ref KAGGLE : dataset introuvable, run
# mort, quota brûlé pour rien.
#   python3 stt_push.py push stt_kernel_v10.py khidmeti-stt-retrain-93h
# ══════════════════════════════════════════════════════════════════════════════
import csv, gc, json, os, random, re, subprocess, sys, time

T0 = time.time()
DEADLINE_S = 7.6 * 3600          # ponytail: budget mur ; au-delà on exporte quand même

os.environ["TOKENIZERS_PARALLELISM"] = "false"
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "datasets==3.6.0", "soundfile", "librosa", "jiwer",
                "onnx", "onnxruntime"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

import numpy as np
import soundfile as sf
import torch
import jiwer
from datasets import load_dataset, Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, \
    get_linear_schedule_with_warmup
from huggingface_hub import snapshot_download, HfApi

SEED = 20260817
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

HF_TOKEN = "{{HF_TOKEN}}"
os.environ["HF_TOKEN"] = HF_TOKEN
BASE_MODEL = "boumehdi/wav2vec2-large-xlsr-moroccan-darija"
REPO       = "Walidrbh27/khidmeti-stt-ctc"        # repo servi si le gate passe
CAND_REPO  = "Walidrbh27/khidmeti-stt-ctc-cand"   # filet : poids sauvés avant les evals
CKPT_REPO  = "Walidrbh27/khidmeti-stt-ckpt"       # resume multi-session (93h)
# Le repo CKPT vit sur le namespace HF (Walidrbh27), comme ses frères ci-dessus,
# JAMAIS sur le compte Kaggle qui exécute : ce sont deux comptes distincts
# (Kaggle walidrbh27khidmeti ≠ HF Walidrbh27) et le seul token embarqué ici est
# celui de HF. Bug mesuré le 24/08 : CKPT_REPO templaté en {{KAGGLE_USERNAME}}
# → « 403 : You don't have the rights to create a model under the namespace
# walidrbh27khidmeti » après 25 min de prep (v10 extrait, 90,5 h chargées).
# Corollaire utile : un repo ckpt unique laisse une session tourner sur le quota
# d'un AUTRE compte Kaggle et reprendre quand même où on s'était arrêté.
# Env de reprise : chaque session Kaggle a un mur à 9 h — 90 h d'audio ≈ 3 h/
# époque → une session ne fait que ~3 époques B. Sans checkpoint, 3 sessions
# ne valent RIEN de plus qu'une. On sauve l'état (meilleur B + vocab + epoch)
# après chaque époque, on reprend au boot. La session 1 (ce push) repart du
# socle ; les pushes suivants (même kernel, reset_done lu depuis le ckpt)
# continuent où la précédente s'est arrêtée. Le GATE reste identique : le
# dernier push évalue, les poids servis sont le meilleur B GLOBAL de toutes les
# sessions, pas seulement de la dernière.
RESUME = os.environ.get("RESUME", "1") != "0"   # "0" = repartir du socle exprès

# Constante MESURÉE par le run apparié `khidmeti-stt-paired` : whisper servi
# en réglages de PRODUCTION (language=ar beam=1 vad=True prompt=True, int8
# 4 threads, 831 clips) = 0,6097 IC95 [0,5930 ; 0,6276]. L'ancienne 0,6300
# venait d'une estimation SANS vad+prompt : trop laxiste de ~2 pts.
SERVED_WER, SERVED_CER = 0.6097, 0.2368   # whisper medium+LoRA int8, même test 831
ZS_LARGE_V3_WER        = 0.8055           # whisper-large-v3 nu, même test
GREEDY_V8_WER          = 0.6914           # v8 greedy int8, mesuré en v9a
LM_V9_WER, LM_V9_CER   = 0.6505, 0.2444   # v8 + LM 4-gram faisceau 16, mesuré v9a

# EP_B ramené de 16 à 8 : à 24 h d'audio une époque coûte ~50 min (mesuré :
# 12 min pour 6 h) → 8 époques SATURENT le budget, et le garde-fou de deadline
# exporte de toute façon. v10 atteignait son meilleur à l'époque 11 avec 4×
# moins de données uniques : le nombre de PAS vus reste comparable.
EP_A, EP_B      = 1, 8                    # étage A (darija générique), B (algérien)
PATIENCE        = 3                       # arrêt sur plateau réel (leçon v8)
SERVED_MODEL    = "Walidrbh27/khidmeti-stt"  # whisper servi : latence CPU de référence
LR_A, LR_B      = 1e-4, 3e-5
BATCH, ACCUM    = 2, 8                    # RAM T4 : ne pas monter sans mesure
MAX_DUR_TRAIN   = 20.0                    # >20 s : hors entraînement (RAM T4)
AUX_DVOICE      = 2500                    # plafond (temps mur, pas dogme)
DEV = "cuda"

# ══ 1b. checkpoint resume (multi-session — le mur ~9 h de Kaggle) ═════════════
# Une session GPU ≈ 7,6 h utiles → ~3 époques de B sur 90 h. Le resume permet
# de chaîner : ckpt torch UNIQUE (state du MEILLEUR B + vocab + compteurs),
# uploadé sur un repo dédié à la FIN de chaque session (marge 70 min de la
# deadline) ; au boot, s'il existe, on le charge et on continue. Le best est
# GLOBAL à toutes les sessions (chargé AVANT l'étage B) → patience et gate
# portent sur l'ensemble, pas sur la dernière session.
CKPT_FN = "ckpt_v90.pt"
# Fail-fast namespace/permission : AU BOOT (seconde ~20), pas à la minute 25.
# Le 24/08, ce create_repo vivait dans load_ckpt() appelé après la prep (v10
# extrait, 90,5 h chargées, socle chargé) → le 403 a coûté 25 min de T4 pour une
# faute d'une ligne. Ici, un namespace ou un token faux tue la session tout de
# suite. exist_ok=True → no-op sur les sessions 2..n.
HfApi(token=HF_TOKEN).create_repo(CKPT_REPO, private=True, exist_ok=True)

def save_ckpt(epoch_b, state, best_wer, sessions):
    """Écrit + upload le ckpt de fin de session. state = state_dict CPU (best)."""
    local = "/kaggle/working/" + CKPT_FN
    torch.save({"epoch_b": epoch_b, "best_wer": best_wer,
                "state": state, "sessions": sessions}, local)
    HfApi(token=HF_TOKEN).upload_file(
        path_or_fileobj=local, path_in_repo=CKPT_FN, repo_id=CKPT_REPO)
    print(f"CKPT_SAVED epoch_b={epoch_b} best_wer={best_wer:.4f} "
          f"sessions={sessions} → {CKPT_REPO}", flush=True)

def load_ckpt():
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(CKPT_REPO, CKPT_FN, token=HF_TOKEN)
        c = torch.load(p, map_location="cpu")
        print(f"CKPT_LOADED epoch_b={c['epoch_b']} best_wer={c['best_wer']:.4f} "
              f"sessions={c['sessions']}", flush=True)
        return c
    except Exception as e:
        print(f"NO_CKPT ({type(e).__name__}: {str(e)[:120]}) — départ du socle",
              flush=True)
        return None

# ---- normalisation arabe pour WER (IDENTIQUE v3/v6/v7/v8/v9 — comparabilité) ----
_DIAC = re.compile(r"[ً-ْٰـ]")
_PUNC = re.compile(r"[^\w\s؀-ۿ]|_")
def norm(s):
    s = re.sub(r"<[^>]*>|\[[^\]]*\]", " ", s)
    s = _DIAC.sub("", s)
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    s = s.replace("ة", "ه").replace("ى", "ي")
    s = _PUNC.sub(" ", s.lower())
    return " ".join(s.split())

def wer_cer(refs, hyps):
    R = [norm(r) or "-" for r in refs]
    H = [norm(h) for h in hyps]
    return jiwer.wer(R, H), jiwer.cer(R, H)

def left(): return DEADLINE_S - (time.time() - T0)

assert norm("الأستاذَة  [bruit] ى!") == "الاستاذه ي", norm("الأستاذَة  [bruit] ى!")
assert wer_cer(["خاصني بلومبي"], ["خاصني بلومبي"])[0] == 0.0
assert wer_cer(["خاصني بلومبي"], ["خاصني كهربائي"])[0] == 0.5

# ══ 1. données ════════════════════════════════════════════════════════════════
# Un item = {"text": labels normalisés, "dur": s, + UNE source audio} :
#   "array" (déjà en RAM)  |  "path" (flac sur disque)  |  ("ds","i") (Arrow mmap)
# Seul l'algérien (val/test + train cible) est matérialisé : le reste reste
# paresseux, sinon 5 000 clips en RAM = 2 Go pour rien.
def get_audio(it):
    if "array" in it:
        return it["array"]
    if "path" in it:
        a, _ = sf.read(it["path"], dtype="float32")
        return a if a.ndim == 1 else a.mean(1)
    a = it["ds"][it["i"]]["audio"]["array"]
    return np.asarray(a, dtype=np.float32)

def clean(t):
    return norm(re.sub(r"<[^>]*>|\[[^\]]*\]", " ", t))

print("Loading Casablanca:Algeria …", flush=True)
ds = load_dataset("UBC-NLP/Casablanca", "Algeria")
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

def cas_items(split):
    out = []
    for ex in ds[split]:
        t = clean(ex["transcription"])
        d = float(ex["duration"])
        if 0.5 < d < 29.5 and len(t) > 1:
            out.append({"text": t, "dur": d,
                        "array": np.asarray(ex["audio"]["array"], dtype=np.float32)})
    return out

tr_full = cas_items("validation")     # Casablanca : "validation" = notre train DZ
test    = cas_items("test")
random.shuffle(tr_full)               # ordre IDENTIQUE v8/v9a (SEED) → même val
n_val = max(1, len(tr_full) // 10)
val, train_dz = tr_full[:n_val], tr_full[n_val:]
assert len(test) == 831, f"test différent des runs v3/v7 : {len(test)}"   # comparabilité
print(f"dz train={len(train_dz)} val={len(val)} test={len(test)}", flush=True)

# --- aux étage A : DVoice v2.0 Darija (le corpus du socle, cc-by-4.0) ---------
# datasets 3.x ne charge plus les scripts de dataset (darija20.py) : on lit
# metadata.csv et on tire les flac directement, seul le sous-ensemble retenu.
aux = []
try:
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import hf_hub_download
    import csv as _csv
    md = hf_hub_download("BrunoHays/DVOICEv2.0-Darija", "metadata.csv",
                         repo_type="dataset", token=HF_TOKEN)
    rows, seen = [], set()
    with open(md, encoding="utf-8") as f:
        for r in _csv.DictReader(f):    # colonnes vérifiées : file_name,words,duration,audio_id
            t, d = clean(r["words"]), float(r["duration"])
            if 0.7 < d < MAX_DUR_TRAIN and len(t) > 1:
                rows.append((r["file_name"], t, d,
                             "-".join(r["audio_id"].split("-")[-5:])))   # <aug>-<uuid> → uuid
    random.shuffle(rows)
    # v2.0 = surtout des copies augmentées de la même phrase : on garde UNE
    # variante par énoncé, sinon l'étage A tourne 4× sur le même contenu.
    uniq = []
    for r in rows:
        if r[3] not in seen:
            seen.add(r[3]); uniq.append(r)
    rows = uniq[:AUX_DVOICE]
    print(f"dvoice: {len(uniq)} énoncés uniques → {len(rows)} tirés", flush=True)
    def pull(r):
        try:
            return hf_hub_download("BrunoHays/DVOICEv2.0-Darija", r[0],
                                   repo_type="dataset", token=HF_TOKEN), r[1], r[2]
        except Exception:
            return None
    with ThreadPoolExecutor(16) as pool:
        for got in pool.map(pull, rows):
            if got:
                aux.append({"text": got[1], "dur": got[2], "path": got[0]})
    print(f"dvoice={len(aux)} clips ({sum(a['dur'] for a in aux)/3600:.1f} h)", flush=True)
except Exception as e:
    print(f"WARN_NO_DVOICE: {type(e).__name__}: {str(e)[:200]}", flush=True)

# --- aux étage A : Casablanca:Morocco (conversationnel, déjà utilisé en v3/v7) -
try:
    ma = load_dataset("UBC-NLP/Casablanca", "Morocco")["validation"]
    ma_meta = ma.remove_columns(["audio"])          # pas de décodage pour filtrer
    ma = ma.cast_column("audio", Audio(sampling_rate=16000))
    n0 = len(aux)
    for i, ex in enumerate(ma_meta):
        t, d = clean(ex["transcription"]), float(ex["duration"])
        if 0.5 < d < MAX_DUR_TRAIN and len(t) > 1:
            aux.append({"text": t, "dur": d, "ds": ma, "i": i})
    print(f"casablanca_ma={len(aux)-n0} clips", flush=True)
except Exception as e:
    print(f"WARN_NO_MOROCCO_AUX: {type(e).__name__}: {str(e)[:200]}", flush=True)

train_b = [x for x in train_dz if x["dur"] <= MAX_DUR_TRAIN] * 2   # DZ ×2 (recette v3/v7)

# --- étage B : +56 k clips v10 (cuits par stt_prep_v10.py, 24 h d'audio) -------
# Transport HF (le multipart Kaggle coupe en EOF TLS depuis un kernel — v13).
# Le prep pousse v10.zip sur le dataset privé HF Walidrbh27/khidmeti-stt-v10 ;
# ici : snapshot_download (cache disque, braqué sur son propre dossier pour ne
# pas polluer HF_HOME par défaut) puis extraction zip.
import zipfile as _zipfile
from huggingface_hub import snapshot_download as _snap

V10 = "/kaggle/working/v10"
# Cache SEUL, pas de local_dir : avec les deux, hub duplique le zip (cache +
# copie locale) → pic 18 Go sur 90 h (6×3) et un disque de 19,5 Go explose au
# chargement. Cache seul : 6 (zip) + 6 (flac extraits) = 12 Go de pic.
v10snap = _snap("Walidrbh27/khidmeti-stt-v10", repo_type="dataset",
                token=HF_TOKEN, allow_patterns=["v10.zip"],
                cache_dir="/kaggle/working/hf_cache_v10")
zp = os.path.join(v10snap, "v10.zip")
assert os.path.isfile(zp), f"v10.zip introuvable dans {v10snap}"
with _zipfile.ZipFile(zp) as zf:
    zf.extractall(V10)                      # → metadata.csv + audio/*.flac + corpus.txt
import shutil as _sh
_sh.rmtree("/kaggle/working/hf_cache_v10", ignore_errors=True)
print(f"v10 extrait dans {V10} (cache libéré)", flush=True)
big56 = []
with open(os.path.join(V10, "metadata.csv"), newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        t, d = clean(r["text"]), float(r["duration"])
        if 0.5 < d <= MAX_DUR_TRAIN and len(t) > 1:
            big56.append({"text": t, "dur": d, "path": os.path.join(V10, r["file_name"])})
# ponytail: sf.read de chemin flac → soundfile décode nativement, aucune lib en plus
assert big56, "metadata.csv v10 vide"
hours56 = sum(x["dur"] for x in big56) / 3600.0
assert 80.0 <= hours56 <= 95.0, f"heures v10 inattendues : {hours56:.1f}"

pool = train_b + big56
print(f"étage A={len(aux)} clips  étage B={len(pool)} clips "
      f"(DZ ×2={len(train_b)} + v10={len(big56)} ≈ {hours56:.1f} h d'audio ; "
      f"DZ >{MAX_DUR_TRAIN:.0f}s écartés: {len(train_dz)-len(train_b)//2})", flush=True)
assert len(pool) > 500, "trop peu de données algériennes — abandon"

# ══ 2. vocabulaire : vérifié AVANT d'entraîner (leçon v6) ═════════════════════
processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL)
tok, fx = processor.tokenizer, processor.feature_extractor
BLANK = tok.pad_token_id
vocab = tok.get_vocab()
chars = {c for it in (pool + aux) for c in it["text"]} - {" "}
missing = sorted(chars - set(vocab))
print(f"vocab={len(vocab)} blank={BLANK} caractères corpus={len(chars)} "
      f"hors-vocab={missing}", flush=True)
if missing:
    tok.add_tokens(missing)          # ids ajoutés à la fin → lm_head élargi plus bas
    vocab = tok.get_vocab()
    print(f"vocab étendu → {len(vocab)}", flush=True)

# aller-retour tokenizer : group_tokens=False, sinon le décodage CTC écrase les
# lettres doublées («الله») et l'assert échouerait pour la mauvaise raison.
for it in random.sample(pool + aux, min(200, len(pool + aux))):
    back = tok.decode(tok(it["text"]).input_ids, group_tokens=False)
    assert back == it["text"], f"aller-retour cassé: {it['text']!r} → {back!r}"

ID2TOK = [""] * (max(vocab.values()) + 1)
for t, i in vocab.items():
    ID2TOK[i] = t
SPECIAL = {tok.pad_token, tok.unk_token, tok.bos_token, tok.eos_token} - {None}

def ctc_greedy(ids):
    """Décodage de service — REJOUÉ TEL QUEL dans docker/ai-stt/server.py."""
    out, prev = [], -1
    for i in ids:
        if i != prev and i != BLANK:
            t = ID2TOK[i]
            if t not in SPECIAL:
                out.append(t)
        prev = i
    return " ".join("".join(out).replace("|", " ").split())

A, B = vocab["م"], vocab["ل"]
assert ctc_greedy([BLANK, A, A, BLANK, 2, BLANK, B, B]) == f"{ID2TOK[A]} {ID2TOK[B]}"
assert ctc_greedy([A, BLANK, A]) == ID2TOK[A] * 2, "lettre doublée écrasée"

# parité feature extractor : ce que server.py devra faire à la main (do_normalize)
_probe = np.random.default_rng(0).normal(0, 0.2, 16000).astype(np.float32)
_ref = fx(_probe, sampling_rate=16000, return_tensors="np").input_values[0]
_mine = (_probe - _probe.mean()) / np.sqrt(_probe.var() + 1e-7)
assert np.abs(_ref - _mine).max() < 1e-4, "normalisation serveur ≠ feature extractor"

# ══ 3. modèle ═════════════════════════════════════════════════════════════════
model = Wav2Vec2ForCTC.from_pretrained(BASE_MODEL, ctc_loss_reduction="mean",
                                       ctc_zero_infinity=True)   # sinon un clip
                                       # trop court pour son label = loss inf = run mort
if missing:
    old = model.lm_head
    new = torch.nn.Linear(old.in_features, len(ID2TOK))
    with torch.no_grad():
        new.weight.zero_(); new.bias.fill_(-5.0)     # nouveaux caractères improbables au départ
        new.weight[:old.out_features] = old.weight
        new.bias[:old.out_features] = old.bias
    model.lm_head = new
    model.config.vocab_size = len(ID2TOK)
model.config.apply_spec_augment = True
model.config.mask_time_prob, model.config.mask_feature_prob = 0.05, 0.02
model.freeze_feature_encoder()          # frontend conv gelé : recette XLSR standard
model.gradient_checkpointing_enable()
model.to(DEV)
print(f"params entraînables: "
      f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.0f} M", flush=True)

def collate(items):
    feats = fx([get_audio(it) for it in items], sampling_rate=16000,
               return_tensors="pt", padding=True)
    enc = tok([it["text"] for it in items], return_tensors="pt", padding=True)
    labels = enc.input_ids.masked_fill(enc.attention_mask.ne(1), -100)
    return feats.input_values, feats.get("attention_mask"), labels

@torch.no_grad()
def torch_eval(items, tag):
    model.eval(); hyps = []
    for k, it in enumerate(items):
        x = fx(get_audio(it), sampling_rate=16000,
               return_tensors="pt").input_values.to(DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            ids = model(x).logits[0].argmax(-1).tolist()
        hyps.append(ctc_greedy(ids))
        if k == 0:
            print(f"[{tag}] hyp0: {hyps[0][:120]}", flush=True)
    w, c = wer_cer([it["text"] for it in items], hyps)
    print(f"{tag}_WER={w:.4f} {tag}_CER={c:.4f}", flush=True)
    return w, c, hyps

# ══ 4. entraînement 2 étages ══════════════════════════════════════════════════
best = {"wer": float("inf"), "cer": None, "tag": "base", "state": None}
skipped, stopped = 0, ""
g_epoch_b = 0                     # époque B courante (multi-session)

def keep_if_best(w, c, tag):
    if w < best["wer"]:
        best.update(wer=w, cer=c, tag=tag,
                    state={k: v.detach().cpu().clone()
                           for k, v in model.state_dict().items()})
        print(f"NEW_BEST={tag} val_WER={w:.4f}", flush=True)

def stage(items, epochs, lr, tag, epoch_start=0):
    global skipped, g_epoch_b
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=lr, weight_decay=0.01)
    steps = max(1, len(items) // (BATCH * ACCUM)) * (epochs - epoch_start)
    sched = get_linear_schedule_with_warmup(opt, max(1, int(steps * 0.1)), steps)
    scaler = torch.amp.GradScaler("cuda")
    order = list(range(len(items)))
    bad = 0
    for ep in range(epoch_start, epochs):
        if tag == "B":
            g_epoch_b = ep + 1    # visible du ckpt de fin de session
        model.train(); random.shuffle(order); tot, nb, stop = 0.0, 0, ""
        opt.zero_grad()
        for j in range(0, len(order), BATCH):
            xv, am, labs = collate([items[k] for k in order[j:j + BATCH]])
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_values=xv.to(DEV),
                             attention_mask=None if am is None else am.to(DEV),
                             labels=labs.to(DEV)).loss / ACCUM
            if not torch.isfinite(loss):   # fp16 + CTC : un step pourri ne tue pas le run
                skipped += 1; opt.zero_grad(); continue
            scaler.scale(loss).backward()
            tot += loss.item() * ACCUM; nb += 1
            if (j // BATCH) % ACCUM == ACCUM - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
            if left() < 4200:              # 70 min : export + upload candidat + evals
                stop = f"deadline_{tag}_ep{ep+1}_step{j}"; break
        w, c, _ = torch_eval(val, f"val_{tag}_ep{ep+1}")
        print(f"[{tag}] epoch {ep+1} loss={tot/max(nb,1):.4f} val_WER={w:.4f} "
              f"skipped={skipped} left={left()/3600:.1f}h", flush=True)
        improved = w < best["wer"]
        keep_if_best(w, c, f"{tag}_ep{ep+1}")
        if stop:
            print(f"DEADLINE_STOP={stop}", flush=True); return stop
        bad = 0 if improved else bad + 1
        if bad >= PATIENCE:      # plateau réel, pas un plafond arbitraire (leçon v8)
            print(f"EARLY_STOP={tag}_ep{ep+1} après {PATIENCE} époques sans gain",
                  flush=True)
            return ""
    return ""

# ══ 4b. reprise multi-session : charger le ckpt AVANT toute décision d'étage ══
# RESUME=0 (env, jamais codé en dur) → repartir du socle, ckpt ignoré.
ck = None
if RESUME:
    ck = load_ckpt()
    if ck is not None:
        # Le lm_head a la taille du vocab ÉTENDU (même donnée à chaque boot →
        # même taille). Une reprise dans un vocab différent = état corrompu :
        # vérifier PLUTÔT que d'importer silencieusement un état incohérent.
        got = model.load_state_dict(ck["state"], strict=True)
        assert not got.missing_keys and not got.unexpected_keys, (
            f"resume incompatible: {got}")
        best["wer"], best["tag"] = ck["best_wer"], "resume"
        # L'état repris EST le best courant : si AUCUNE époque de cette session
        # ne l'améliore, best["state"] doit quand même exister — sinon l'assert
        # de fin tuait la session APRÈS le training, avant tout export (bug
        # potentiel qui aurait gaspillé 7,6 h).
        best["state"] = {k: v.detach().cpu().clone()
                         for k, v in model.state_dict().items()}
        g_epoch_b = ck["epoch_b"]
        print(f"RESUME_STATE best_wer={best['wer']:.4f} epoch_b={g_epoch_b}",
              flush=True)

print(f"=== socle nu (zero-shot darija marocain) — {left()/3600:.1f} h ===", flush=True)
zs_val = torch_eval(val, "ZS_BASE_val")
if ck is not None:
    # Vérification gratuite du resume : le val WER mesuré doit retomber sur le
    # best_wer stocké (même découpe, SEED identique). Un écart > 0.02 = reprise
    # cassée → le savoir à la minute 10, pas après 7 h.
    _drift = abs(zs_val[0] - ck["best_wer"])
    print(f"RESUME_CHECK val_mesuré={zs_val[0]:.4f} vs ckpt={ck['best_wer']:.4f} "
          f"drift={_drift:.4f}" + (" OK" if _drift < 0.02 else " ⚠️ RESUME DOUTEUX"),
          flush=True)
    assert _drift < 0.05, f"resume cassé: drift {_drift:.4f}"
    # L'état repris EST le best courant : son CER, mesuré à l'instant, complète
    # best["cer"] que le ckpt ne stocke pas. Sans ça, une session où AUCUNE
    # époque n'améliore (plateau — le cas dès qu'on approche EP_B) laisse
    # best["cer"]=None, et le meta.json final meurt sur round(None) APRÈS
    # l'export et les evals : 5,2 h de T4 jetées à la dernière ligne (25/08,
    # sessions v5 puis v7, deux fois le même mur).
    best["cer"] = zs_val[1]
if aux and g_epoch_b == 0:        # étage A déjà fait dans une session précédente
    print(f"=== étage A : darija générique {len(aux)} clips ===", flush=True)
    stopped = stage(aux, EP_A, LR_A, "A")
if not stopped:
    print(f"=== étage B : algérien {len(pool)} clips (≈{hours56:.0f} h v10 + DZ ×2) "
          f"— reprend époque {g_epoch_b}/{EP_B} ===", flush=True)
    stopped = stage(pool, EP_B, LR_B, "B", epoch_start=g_epoch_b)
assert best["state"] is not None, "aucun checkpoint validé"
model.load_state_dict(best["state"])
print(f"BEST={best['tag']} val_WER={best['wer']:.4f} (socle nu {zs_val[0]:.4f})", flush=True)
if best["wer"] > zs_val[0]:      # le fine-tune a dégradé le socle : à savoir AVANT le gate
    print(f"WARN_FT_WORSE_THAN_BASE: {best['wer']:.4f} > {zs_val[0]:.4f}", flush=True)

# Fin de session : sauver l'état pour la PROCHAINE (jamais la dernière ligne —
# la deadline laisse 70 min, l'upload <50 Mo est instantané). best["state"] est
# déjà CPU (cloné dans keep_if_best) — PAS de transfert GPU/CPU à la deadline.
save_ckpt(g_epoch_b, best["state"], best["wer"], (ck or {}).get("sessions", 0) + 1)

# ══ 5. export ONNX (moteur de service) ════════════════════════════════════════
model = model.cpu().float().eval()
model.gradient_checkpointing_disable()      # torch.utils.checkpoint casse le tracing
model.config.apply_spec_augment = False
del best["state"]; gc.collect(); torch.cuda.empty_cache()

class CtcLogits(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, input_values):
        return self.m(input_values=input_values).logits

os.makedirs("out", exist_ok=True)
torch.onnx.export(
    CtcLogits(model).eval(), (torch.zeros(1, 32000),), "out/model.fp32.onnx",
    input_names=["input_values"], output_names=["logits"],
    dynamic_axes={"input_values": {0: "batch", 1: "samples"},
                  "logits": {0: "batch", 1: "frames"}},
    opset_version=17, dynamo=False,          # leçon P2 : dynamo exige onnxscript
)
from onnxruntime.quantization import QuantType, quantize_dynamic
# MatMul seulement : en v8, quantifier aussi les Conv du feature encoder donnait
# un int8 plus mauvais (0,7311 vs 0,6977) ET plus lent (0,469 vs 0,268 ×RT).
quantize_dynamic("out/model.fp32.onnx", "out/model.int8.onnx",
                 weight_type=QuantType.QInt8, per_channel=True,
                 op_types_to_quantize=["MatMul"])
json.dump({"id2tok": ID2TOK, "blank_id": BLANK, "delim": tok.word_delimiter_token,
           "special": sorted(SPECIAL), "sampling_rate": 16000},
          open("out/ctc_vocab.json", "w"), ensure_ascii=False)
print("exporté:", {f: round(os.path.getsize("out/" + f) / 1e6) for f in os.listdir("out")},
      flush=True)

import onnxruntime as ort
_OPT = ort.SessionOptions(); _OPT.intra_op_num_threads = 4   # = STT_THREADS en prod

def onnx_eval(fname, items, tag):
    s = ort.InferenceSession("out/" + fname, _OPT, providers=["CPUExecutionProvider"])
    hyps, t0 = [], time.time()
    for k, it in enumerate(items):
        x = fx(get_audio(it), sampling_rate=16000,
               return_tensors="np").input_values.astype(np.float32)
        hyps.append(ctc_greedy(s.run(None, {"input_values": x})[0][0].argmax(-1).tolist()))
        if k == 0:
            print(f"[{tag}] hyp0: {hyps[0][:120]}", flush=True)
    w, c = wer_cer([it["text"] for it in items], hyps)
    rt = (time.time() - t0) / max(1e-9, sum(it["dur"] for it in items))
    print(f"{tag}_WER={w:.4f} {tag}_CER={c:.4f} xRT_cpu={rt:.3f}", flush=True)
    return w, c, rt
# smoke 3 clips : un export cassé se voit en 1 min, pas après 6 h (leçon v7)
assert onnx_eval("model.fp32.onnx", test[:3], "SMOKE")[0] < 1.5, "export ONNX suspect"

# FILET : poids sur HF AVANT les evals longues (leçon v7 : 6 h perdues comme ça)
api = HfApi(token=HF_TOKEN)
api.create_repo(CAND_REPO, private=True, exist_ok=True)
api.upload_folder(folder_path="out", repo_id=CAND_REPO)
print(f"CANDIDATE_SAVED={CAND_REPO} (best_val_WER={best['wer']:.4f})", flush=True)

# ══ 6. evals finales ══════════════════════════════════════════════════════════
# Le test complet en ONNX/CPU coûterait ~45 min (831 clips, 2,5 h d'audio). On
# mesure donc : test complet en torch GPU (chiffre de tête) + sous-ensemble fixe
# de 150 clips dans LES TROIS moteurs (torch / onnx fp32 / onnx int8) pour
# quantifier l'écart de quantification, et on gate sur l'estimation servie.
print(f"=== evals finales — {left()/3600:.1f} h ===", flush=True)
model.to(DEV)
full_wer, full_cer, full_hyps = torch_eval(test, "FT_CTC_TEST_TORCH")
model.cpu(); gc.collect(); torch.cuda.empty_cache()

sub = random.Random(SEED).sample(range(len(test)), min(150, len(test)))
sub_items = [test[i] for i in sub]
sub_torch = wer_cer([it["text"] for it in sub_items], [full_hyps[i] for i in sub])[0]
f32_wer, f32_cer, f32_rt = onnx_eval("model.fp32.onnx", sub_items, "SUB_ONNX_FP32")
i8_wer, i8_cer, i8_rt = onnx_eval("model.int8.onnx", sub_items, "SUB_ONNX_INT8")
print(f"SUB_TORCH_WER={sub_torch:.4f}", flush=True)

# Référence de VITESSE : le whisper réellement servi, même CPU 4 threads, mêmes
# clips. Sans ce chiffre, «plus léger / plus rapide» reste une supposition. Bloc
# non fatal : il arrive après le gate, il ne peut pas coûter le run.
wh_rt = wh_wer = None
wh_items = sub_items[:60]                 # borne le temps mur (whisper ≈ 1×RT CPU)
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "faster-whisper"],
                   check=True)
    from faster_whisper import WhisperModel
    wm = WhisperModel(SERVED_MODEL, device="cpu", compute_type="int8", cpu_threads=4)
    hyps, t0 = [], time.time()
    for it in wh_items:
        segs, _ = wm.transcribe(get_audio(it), language="ar", beam_size=1)
        hyps.append(" ".join(s.text for s in segs).strip())
    wh_rt = (time.time() - t0) / max(1e-9, sum(it["dur"] for it in wh_items))
    wh_wer = wer_cer([it["text"] for it in wh_items], hyps)[0]
    print(f"SUB60_WHISPER_SERVI_WER={wh_wer:.4f} xRT_cpu={wh_rt:.3f}", flush=True)
    del wm; gc.collect()
except Exception as e:
    print(f"WARN_WHISPER_BASELINE_SKIPPED: {type(e).__name__}: {e}", flush=True)

# int8 servi seulement s'il ne coûte rien de mesurable (règle de vision_kernel)
serve_file = "model.int8.onnx" if i8_wer <= f32_wer + 0.01 else "model.fp32.onnx"
served_gap = (i8_wer if serve_file.endswith("int8.onnx") else f32_wer) - sub_torch
wer_served_est = full_wer + max(0.0, served_gap)
print(f"serve_file={serve_file} écart_moteur={served_gap:+.4f} "
      f"→ WER servi estimé {wer_served_est:.4f}", flush=True)

bins = {}
for lo, hi in [(0.5, 5), (5, 10), (10, 20), (20, 30)]:
    idx = [i for i, it in enumerate(test) if lo < it["dur"] <= hi]
    if len(idx) >= 10:
        w, c = wer_cer([test[i]["text"] for i in idx], [full_hyps[i] for i in idx])
        bins[f"{lo}-{hi}s"] = {"n": len(idx), "wer": round(w, 4), "cer": round(c, 4)}

gate_pass = wer_served_est < SERVED_WER
print("\n┌─ récap (mêmes 831 clips Casablanca-Algeria) ─")
print(f"│ whisper medium+LoRA SERVI (CT2 int8) ... WER {SERVED_WER:.4f}")
print(f"│ whisper-large-v3 nu ................... WER {ZS_LARGE_V3_WER:.4f}")
print(f"│ CTC v8 greedy (int8, v9a) ............. WER {GREEDY_V8_WER:.4f}")
print(f"│ CTC v8 + LM 4-gram (int8, v9a) ........ WER {LM_V9_WER:.4f} CER {LM_V9_CER:.4f}")
print(f"│ wav2vec2 darija nu (val) .............. WER {zs_val[0]:.4f}")
print(f"│ wav2vec2 darija v10 (torch, test) ..... WER {full_wer:.4f} CER {full_cer:.4f}")
print(f"│ wav2vec2 darija v10 (servi, estimé) ... WER {wer_served_est:.4f}")
print(f"│ vitesse CPU 4 threads ................. fp32 {f32_rt:.3f}×RT  int8 {i8_rt:.3f}×RT")
if wh_rt:
    print(f"│ whisper servi, MÊME CPU (60 clips) .... {wh_rt:.3f}×RT  WER {wh_wer:.4f}")
    print(f"│ gain de vitesse CTC vs whisper ........ ×{wh_rt / min(f32_rt, i8_rt):.1f}")
print(f"│ WER par durée ......................... {bins}")
print("└─", flush=True)
print(f"GATE_BEATS_SERVED={'YES' if gate_pass else 'NO'} "
      f"({wer_served_est:.4f} vs {SERVED_WER:.4f})")
print(f"GATE_PASS={'YES' if gate_pass else 'NO'}", flush=True)

# ══ 7. package + upload ═══════════════════════════════════════════════════════
json.dump({
    "version": "v10-ctc", "trained": "2026-08-18", "seed": SEED,
    "arch": "Wav2Vec2ForCTC (encodeur seul, décodage CTC glouton)",
    "base_model": BASE_MODEL,
    "why_this_base": "déjà pré-entraîné sur la darija (DVoice) contrairement à whisper "
                     "(large-v3 nu = 0,8055 sur ce test : le socle whisper n'apporte "
                     "rien en darija, tout venait du LoRA) ; CTC = une passe, coût "
                     "proportionnel à la durée réelle, pas de fenêtre 30 s à remplir",
    "method": f"2 étages : A darija générique {EP_A} ép lr {LR_A}, B algérien "
              f"({len(pool)} items ≈ {hours56:.0f} h v10 + DZ ×2) jusqu'à {EP_B} ép "
              f"lr {LR_B} (arrêt sur plateau, patience {PATIENCE}) ; feature encoder "
              "gelé, spec-augment, fp16, clip 1.0",
    "hparams": {"batch": BATCH, "accum": ACCUM, "max_dur_train": MAX_DUR_TRAIN,
                "best": best["tag"], "skipped_nonfinite_steps": skipped,
                "deadline_stop": stopped or None, "vocab_added": missing},
    "datasets": {"train_dz": "UBC-NLP/Casablanca:Algeria validation ×2 (CC BY-NC-ND 4.0)",
                 "v10": "56 k clips algériens vérifiés, cuits par stt_prep_v10.py "
                        "(kahwa 23k + algerian_tts 23k + rawi 5k + cafe-codeswitch 3.8k) "
                        "— licences à revalider (oddadmix 'other')",
                 "aux": "BrunoHays/DVOICEv2.0-Darija (CC BY 4.0) + Casablanca:Morocco",
                 "test": "UBC-NLP/Casablanca:Algeria test, 831 clips utilisables"},
    "license_note": "poids intérimaires non-commerciaux (Casablanca CC BY-NC-ND + "
                    "oddadmix 'other') — remplacer par les données flywheel avant "
                    "commercialisation",
    "rows": {"train_dz": len(train_dz), "train_b": len(train_b), "v10": len(big56),
             "dvoice": sum(1 for a in aux if "path" in a),
             "casablanca_ma": sum(1 for a in aux if "ds" in a),
             "val": len(val), "test": len(test)},
    "wer_norm": {"served_whisper_v3": SERVED_WER, "zeroshot_whisper_large_v3": ZS_LARGE_V3_WER,
                 "v8_greedy_int8": GREEDY_V8_WER, "v9a_lm_4gram": LM_V9_WER,
                 "zeroshot_base_val": round(zs_val[0], 4), "best_val": round(best["wer"], 4),
                 "test_torch": round(full_wer, 4), "test_served_est": round(wer_served_est, 4),
                 "sub_torch": round(sub_torch, 4), "sub_onnx_fp32": round(f32_wer, 4),
                 "sub_onnx_int8": round(i8_wer, 4)},
    "cer_norm": {"served_whisper_v3": SERVED_CER, "best_val": round(best["cer"], 4),
                 "test_torch": round(full_cer, 4),
                 "sub_onnx_fp32": round(f32_cer, 4), "sub_onnx_int8": round(i8_cer, 4)},
    "wer_by_duration": bins,
    "xrt_cpu_4threads": {"fp32": round(f32_rt, 4), "int8": round(i8_rt, 4),
                         "whisper_servi_60clips": None if wh_rt is None else round(wh_rt, 4),
                         "whisper_servi_wer_60clips": None if wh_wer is None else round(wh_wer, 4)},
    "serve": "onnxruntime CPU + décodage CTC glouton (STT_ENGINE=ctc)",
    "serve_file": serve_file,
    "decode": "argmax par frame → collapse des répétitions → retrait du blank → "
              "'|' devient espace ; ctc_vocab.json porte id2tok/blank_id/delim",
    "preprocess": "float32 mono 16 kHz, zero-mean/unit-var sur le clip entier "
                  "((x-mean)/sqrt(var+1e-7)) — do_normalize du Wav2Vec2FeatureExtractor",
    "gate": "WER servi estimé < 0.6300 (whisper medium+LoRA int8, même test)",
    "warning_no_vad": "pas de VAD ici : CTC n'hallucine pas sur le silence, "
                      "contrairement à whisper (d'où vad_filter côté whisper)",
}, open("out/meta.json", "w"), indent=2, ensure_ascii=False)

if not gate_pass:
    print(f"GATE_FAILED_NO_UPLOAD — rien ne bouge en prod ; poids sur {CAND_REPO}")
    api.upload_file(path_or_fileobj="out/meta.json", path_in_repo="meta.json",
                    repo_id=CAND_REPO)
    sys.exit(0)

# GATE PASSÉ — mais le train NE PROMEUT JAMAIS vers le repo servi (leçon 20/08 :
# `out/` ne contient ni lm.npz ni la section meta["lm"], donc écraser le servi
# lui ferait perdre les paramètres de décodage mesurés → serveur en config par
# défaut, silencieusement). La promotion appartient au kernel LM, qui re-règle
# le faisceau sur CES poids puis publie la chaîne complète.
print(f"GATE_PASS_CANDIDATE — poids sur {CAND_REPO} (le servi {REPO} est INTACT)")
print("ÉTAPE SUIVANTE OBLIGATOIRE : CPU=1 python3 stt_push.py push "
      "stt_lm_kernel_v9.py khidmeti-stt-lm-v10")
print("  → il re-règle order/alpha/beta/beam sur ces poids, mesure le WER de la "
      "chaîne servie, et publie modèle+lm.npz+meta SEULEMENT si sa mesure passe.")
print(f"V10 COMPLETE: greedy servi estimé {wer_served_est:.4f} "
      f"(whisper prod {SERVED_WER:.4f}) en {(time.time()-T0)/3600:.1f} h")

# Auto-restart Space to pick up new weights
try:
    from huggingface_hub import HfApi as _HfApi2
    _HfApi2(token=HF_TOKEN).restart_space(repo_id="Walidrbh27/khidmeti-ai", repo_type="space")
    print("SPACE_RESTARTED=Walidrbh27/khidmeti-ai")
except Exception as _e:
    print(f"SPACE_RESTART_SKIP={_e}")

