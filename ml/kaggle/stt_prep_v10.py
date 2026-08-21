#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
# ml/kaggle/stt_prep_v10.py — v10, étape 1/2 : cuire les 56k clips algériens.
#
# Contexte mesuré (mêmes 831 clips Casablanca-Algeria, même moteur int8) :
#   whisper medium+LoRA servi ....... WER 0,6300  CER 0,2373
#   CTC v8 (749 énoncés DZ seuls) ... WER 0,6914  CER 0,2531 — loss qui baissait
#     pendant que le val stagnait (0,6675 → 0,6675, 6,8 h GPU non consommées) :
#     SURAPPRENTISSAGE, le plafond de v8 est la DONNÉE (1,5 h d'audio), pas le
#     temps de calcul.
#   CTC v8 + LM 4-gram (v9a) ........ WER 0,6505  — le LM plafonne aussi (corpus
#     ÉCRIT, registre ≠ parlé). Les gains restants sont dans le MODÈLE.
#
# Ce kernel (CPU, quota GPU préservé) matérialise le levier données :
#   56 k clips algériens VÉRIFIÉS (marqueurs dialecte, session v9a) →
#   dédoublonnage (refrains podcasts) → garde-fou anti-fuite (texte exact du
#   test Casablanca) → parts sources (total 24 h d'audio) → flac 16 k + metadata.csv +
#   corpus.txt (registre PARLÉ pour le futur LM) → dataset Kaggle PRIVÉ
#   khidmeti-stt-v10, monté par stt_kernel_v10.py (étape 2, GPU).
#
# Pourquoi pas load_dataset direct dans le kernel d'entraînement : les 4 sources
# font ~30 Go de parquet (audio embarqué), un disque kernel ≈ 19,5 Go n'en
# tiendrait pas la moitié. La sélection se fait ICI, une fois ; le train ne
# télécharge que les ~950 Mo de flac retenus.
#
# Sources (schémas vérifiés datasets-server) :
#   oddadmix/…-kahwa-postcast 23 264 (audio, transcript_text, duration)
#   yasminekaced/algerian_tts  23 356 (audio, text, duration)
#   oddadmix/…-algerian-rawi    5 296 (audio, transcript_text, duration)
#   FatimahEmadEldin/cafe-…     3 830 (audio, transcription — durée par décodage)
#
# Push (CPU) : CPU=1 python3 stt_push.py push stt_prep_v10.py khidmeti-stt-prep-v10
# ══════════════════════════════════════════════════════════════════════════════
import csv, gc, glob, json, os, random, re, shutil, subprocess, sys, time
import traceback
import zipfile

T0 = time.time()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "datasets==3.6.0", "soundfile"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio

SEED = 20260817
BIG_RNG = random.Random(SEED + 1)      # son propre flux : ne touche JAMAIS au RNG
                                       # global (le split val/test de v10 reste
                                       # bit-identique à v8/v9a — comparabilité)
HF_TOKEN = "{{HF_TOKEN}}"
os.environ["HF_TOKEN"] = HF_TOKEN

# Parts cibles PAR SOURCE. Global pas par source : un plafond global laisserait
# kahwa (50 h dispo) manger la part des autres ; ces parts gardent la diversité
# (podcast, télé, contes, code-switch).
# v11 — 24 h au lieu de 6 h. Justification MESURÉE, pas une intuition :
#   v8 (1,5 h) → v10 (6 h) = ×4 de données → greedy 0,6892 → 0,6771 (−0,0121).
#   Et v10 SURAPPRENAIT encore : val 0,6498 à l'époque 11 puis 0,6639 / 0,6686 —
#   le plafond reste la quantité de données uniques, pas le calcul.
#   Coût mesuré : 12 min/époque pour 6 h d'audio (4220 items, T4) → 24 h ≈ 48
#   min/époque → ~8 époques dans le budget, assez (v10 atteignait son meilleur à
#   l'époque 11 avec quatre fois moins de données uniques).
#   Dispo par source : kahwa 50,6 h · dztv 34,0 h · rawi 6,4 h · cafe 1,9 h.
STAGE_B_HOURS = 90.5
# 93h DISPO mesurées (session v9a) : kahwa 50.6 · dztv 34.0 · rawi 6.4 · cafe 1.9.
# On vise 90.5 h (marge dispo) — le vrai plafond est CA PAR SOURCE ci-dessous.
SOURCE_TARGET_HOURS = {"kahwa": 50.0, "dztv": 33.0, "rawi": 6.0, "cafe": 1.5}
MAX_DUR_TRAIN = 20.0        # identique v8 (RAM T4) ; au-delà : corpus LM seulement

WORK = "/kaggle/working"
OUT = f"{WORK}/v10"

SRC56 = [
    # dztv EN PREMIER : source fragile (échec ×2 quand chargée APRÈS kahwa, alors
    # que le probe passe en isolation) → elle charge sur un kernel vierge.
    ("yasminekaced/algerian_tts", "default", "text", "duration", "dztv"),
    ("oddadmix/arabic-audio-collection-algerian-kahwa-postcast", "default",
     "transcript_text", "duration", "kahwa"),
    ("oddadmix/arabic-audio-collection-algerian-rawi", "default",
     "transcript_text", "duration", "rawi"),
    ("FatimahEmadEldin/cafe-algerian-codeswitch-speech", "large",
     "transcription", None, "cafe"),   # pas de colonne durée → décodage à la volée
]
_DZ = ["راني", "تاع", "برك", "وشراك", "بزاف", "علاش", "كيفاش"]
_MA = ["ديال", "غادي", "ماكاين", "واخا"]

# ---- normalisation arabe (IDENTIQUE v3/v6/v7/v8/v9 — comparabilité stricte) ----
_DIAC = re.compile(r"[ً-ْٰـ]")
_PUNC = re.compile(r"[^\w\s؀-ۿ]|_")
def norm(s):
    s = re.sub(r"<[^>]*>|\[[^\]]*\]", " ", s)
    s = _DIAC.sub("", s)
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    s = s.replace("ة", "ه").replace("ى", "ي")
    s = _PUNC.sub(" ", s.lower())
    return " ".join(s.split())

def clean(t):
    return norm(re.sub(r"<[^>]*>|\[[^\]]*\]", " ", t))

def markers(t):
    return sum(t.count(m) for m in _DZ), sum(t.count(m) for m in _MA)

assert norm("الأستاذَة  [bruit] ى!") == "الاستاذه ي"
assert clean("صاڨ  [موسيقى] كيما") == "صاڨ كيما"          # annotations RETIRÉES
assert clean("ڨاع  واش راك؟") == "ڨاع واش راك؟"           # ؟ survit (bloc arabe) — comportement pipeline partagé

# ══ garde-fou anti-fuite : le texte EXACT du test Casablanca dans le train
# ══ trafiquerait le WER mesuré (leçon v9a : fuites_test_jetées).
print("Loading Casablanca:Algeria (test, garde-fou) …", flush=True)
_cs = load_dataset("UBC-NLP/Casablanca", "Algeria", split="test",
                   cache_dir=f"{WORK}/hfcache/test")
test_texts = {t for t in (clean(ex["transcription"]) for ex in _cs) if len(t) > 1}
del _cs; gc.collect()
print(f"test_texts={len(test_texts)}", flush=True)

os.makedirs(f"{OUT}/audio", exist_ok=True)
meta_rows, corpus_lines = [], []
seen, drop_dup, drop_leak = set(), 0, 0

for hf_id, cfg, c_text, c_dur, tag in SRC56:
    cache = f"{WORK}/hfcache/{tag}"
    os.environ["HF_HUB_CACHE"] = cache
    os.environ["HF_DATASETS_CACHE"] = cache
    try:
        print(f"=== {tag}: {hf_id} …", flush=True)
        # v5 : algerian_tts a échoué en DatasetGenerationError (shard flaky ?) —
        # un retry sauve 23 k clips, le skip reste le filet (try/except ci-dessous)
        d = None
        for attempt in (1, 2):
            try:
                d = load_dataset(hf_id, cfg, cache_dir=cache)
                break
            except Exception:
                print(f"  retry {attempt} — TRACEBACK:", flush=True)
                traceback.print_exc()            # leçon v10 : retry sans trace = cause invisible
                gc.collect()
                shutil.rmtree(cache, ignore_errors=True)   # cache semi-téléchargé : on repart propre
                time.sleep(20)
        assert d is not None, "load_dataset: 2 tentatives échouées"
        d = d.cast_column("audio", Audio(sampling_rate=16000))
        n0 = len(d["train"])
        scan = d["train"].remove_columns(["audio"])   # scan SANS décodage audio ;
                                                      # l'audio se tire par index ci-dessous
        keep, audio_cache, dz_n, ma_n = [], {}, 0, 0
        for i, ex in enumerate(scan):
            t = clean(ex[c_text])
            if len(t) < 2:
                continue
            if c_dur:
                dur = float(ex[c_dur])
            else:
                arr = np.asarray(d["train"][i]["audio"]["array"], dtype=np.float32)
                dur = len(arr) / 16000.0
                audio_cache[i] = arr        # café : pas de re-décodage à l'écriture
            dz_, ma_ = markers(t); dz_n += dz_; ma_n += ma_
            if t in test_texts:
                drop_leak += 1; continue
            if (tag, t) in seen:
                drop_dup += 1; continue
            seen.add((tag, t))
            corpus_lines.append(t)          # LM registre parlé : TEXTE seulement, toute durée
            if 0.5 < dur < MAX_DUR_TRAIN:
                keep.append({"i": i, "text": t, "dur": dur})
        print(f"SRC {tag}: scan={n0} garde_avant_plafond={len(keep)} "
              f"duree={sum(x['dur'] for x in keep)/3600:.1f}h "
              f"markers_dz={dz_n} ma={ma_n}", flush=True)
        if ma_n >= dz_n and tag != "cafe":
            print(f"WARN_DIALECTE {tag}: marqueurs MA >= DZ — source suspecte ?", flush=True)

        BIG_RNG.shuffle(keep)               # tirage aléatoire stable du sous-ensemble
        cap = SOURCE_TARGET_HOURS[tag] * 3600
        sel, secs = [], 0.0
        for it in keep:
            # BUG v5 : dur est en SECONDES, STAGE_B_HOURS en heures → le plafond
            # coupait à ~25 s d'audio (1-2 clips). Le run a fini en "trop peu".
            if secs + it["dur"] > cap:
                break
            sel.append(it); secs += it["dur"]
        for k, it in enumerate(sel):
            if it["i"] in audio_cache:
                arr = audio_cache[it["i"]]          # café : décodé au scan
            else:
                arr = np.asarray(d["train"][it["i"]]["audio"]["array"],
                                 dtype=np.float32)
            fname = f"kh-{tag}-{it['i']:06d}.flac"   # flac ≠ wav : ÷3 le zip →
            # transfert upload robuste (EOF TLS vu en v12 sur 690 Mo)
            sf.write(f"{OUT}/audio/{fname}", arr, 16000)
            meta_rows.append({"file_name": f"audio/{fname}",
                              "text": it["text"], "duration": round(it["dur"], 3)})
            if k and k % 5000 == 0:
                print(f"  {k}/{len(sel)} écrits ({secs/3600:.1f} h visées)", flush=True)
        print(f"  → {len(sel)} clips retenus ({secs/3600:.1f} h) écrits", flush=True)
        # scan garde la table Arrow de la source VIVANTE (mmap) : del d seul ne
        # libère rien — c'est le fuite qui a probablement tué dztv chargé après
        # kahwa (2×12,5 Go résidents). Tout ce qui référence la table part.
        del scan, keep, d, audio_cache; gc.collect()
        shutil.rmtree(cache, ignore_errors=True)   # disque 19,5 Go : on libère source par source
    except Exception as e:
        print(f"WARN_SRC56_SKIPPED {hf_id}: {type(e).__name__}: {str(e)[:200]}", flush=True)
        shutil.rmtree(cache, ignore_errors=True)   # cache partiel : libéré aussi

# Plancher en CLIPS : 2728 clips ont donné 6,0 h → 24 h ≈ 11 000 clips. Les
# HEURES (20–26 h) restent le vrai garde-fou du dimensionnement d'époque.
assert len(meta_rows) > 6000, "trop peu de clips — abandon"
assert abs(sum(SOURCE_TARGET_HOURS.values()) - STAGE_B_HOURS) < 1e-9, \
    "parts sources ≠ plafond global"
total_h = sum(float(r["duration"]) for r in meta_rows) / 3600.0
assert 80.0 <= total_h <= 95.0, f"total {total_h:.1f} h hors plage 80–95 h"
with open(f"{OUT}/metadata.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["file_name", "text", "duration"])
    w.writeheader(); w.writerows(meta_rows)
with open(f"{OUT}/corpus.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(corpus_lines) + "\n")
print(f"metadata.csv={len(meta_rows)} corpus={len(corpus_lines)} "
      f"dedup={drop_dup} fuite_test={drop_leak} total={total_h:.1f}h", flush=True)

# ══ self-check : comptes concordants + un flac re-lu mesure sa durée ═════════
wavs = glob.glob(f"{OUT}/audio/*.flac")
assert len(wavs) == len(meta_rows), f"{len(wavs)} flac vs {len(meta_rows)} lignes"
j = BIG_RNG.randrange(len(meta_rows))
a, sr = sf.read(f"{OUT}/{meta_rows[j]['file_name']}")
assert sr == 16000
assert abs(len(a) / 16000 - float(meta_rows[j]["duration"])) <= max(
    0.2, 0.1 * float(meta_rows[j]["duration"])), "durée flac ≠ metadata"
assert corpus_lines and all(corpus_lines), "corpus vide ou lignes vides"

print(f"DATASET_READY {len(meta_rows)} clips {total_h:.1f} h — zip → HF privé …",
      flush=True)

# ══ transport = HF (pas Kaggle) : l'endpoint multipart Kaggle coupe en EOF TLS
# ══ depuis un kernel (v12 : 690 Mo wav, v13 : 410 Mo flac → 3 retries morts),
# ══ alors que le POST via huggingface_hub passe À CETTE ÉCHELLE depuis les
# ══ mêmes kernels (poids ONNX 1,6 Go, uploads v8-v9). Dataset privé HF, monté
# ══ à l'identique par stt_kernel_v10.py (snapshot_download + filenames connues).
HF_REPO = "Walidrbh27/khidmeti-stt-v10"          # privé, créé automatiquement

ZIP = f"{WORK}/v10.zip"
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as zf:   # flac déjà compressé → STORED
    for w in wavs:
        zf.write(w, w[len(OUT) + 1:])
    zf.write(f"{OUT}/metadata.csv", "metadata.csv")
    zf.write(f"{OUT}/corpus.txt", "corpus.txt")

from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)
api.create_repo(HF_REPO, repo_type="dataset", private=True, exist_ok=True)
api.upload_file(path_or_fileobj=ZIP, path_in_repo="v10.zip", repo_id=HF_REPO,
                repo_type="dataset")
print(f"HF_UPLOAD_DONE {HF_REPO} ({os.path.getsize(ZIP)/1e9:.2f} GB)", flush=True)
print(f"PREP_DONE: {len(meta_rows)} clips {total_h:.1f} h en {(time.time()-T0)/60:.1f} min",
      flush=True)
