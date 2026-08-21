#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
# ml/kaggle/stt_paired_v10.py — la comparaison APPARIÉE qui manque.
#
# Où on en est (mesuré, mêmes 831 clips Casablanca-Algeria, même norm) :
#   whisper medium+LoRA « servi » ... WER 0,6300  ← constante du GATE, héritée d'un
#                                      run v3/v4 dont les réglages ne sont PAS ceux
#                                      de docker/ai-stt/server.py (ni vad_filter ni
#                                      initial_prompt). Jamais revérifiée depuis.
#   CTC v10 + LM 4-gram faisceau 32 . WER 0,6389  CER 0,2358  0,300×RT  (×14)
#
# Bootstrap NON APPARIÉ fait à la main sur les hypothèses sauvées (4000 tirages,
# unité = l'énoncé) : IC 95 % du CTC = [0,6223 ; 0,6543]. 0,6300 tombe DEDANS,
# l'écart vaut 1,09 écart-type. Les deux systèmes sont donc à égalité statistique
# — mais un test non apparié est faible : la variance de difficulté des énoncés
# ne se compense pas. Un test APPARIÉ (mêmes énoncés, les deux systèmes) a une
# variance bien plus petite et peut trancher un écart réel de 0,9 pt.
#
# Ce kernel (CPU, aucun GPU) :
#   1. rejoue les 831 clips dans le whisper SERVI, avec les réglages EXACTS de
#      docker/ai-stt/server.py (vad_filter, initial_prompt métier, beam 1, ar) ;
#   2. récupère les hypothèses CTC déjà sauvées (hyps.json du repo candidat) et
#      VÉRIFIE l'alignement énoncé par énoncé via les références ;
#   3. bootstrap APPARIÉ sur la DIFFÉRENCE de WER, plus le compte de victoires
#      par énoncé ;
#   4. dit si l'écart de 0,9 pt est réel ou du bruit, et si la constante 0,6300
#      du gate est encore juste sous les réglages de production.
#
# Sortie : whisper_hyps.json sur le repo candidat → tout futur test apparié est
# gratuit (plus jamais 4 h de whisper CPU).
#
# Push (CPU) : CPU=1 python3 stt_push.py push stt_paired_v10.py khidmeti-stt-paired
# ══════════════════════════════════════════════════════════════════════════════
import json, math, os, random, re, subprocess, sys, time

T0 = time.time()
DEADLINE_S = 8.2 * 3600      # ponytail: au-delà, on publie le partiel — jamais rien

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "datasets==3.6.0", "soundfile", "librosa", "jiwer",
                "faster-whisper", "huggingface_hub"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

import numpy as np
import jiwer
from datasets import load_dataset, Audio
from huggingface_hub import hf_hub_download, HfApi

SEED = 20260817                  # IDENTIQUE v8/v9/v10 → même découpe, même ordre
random.seed(SEED); np.random.seed(SEED)

HF_TOKEN = "{{HF_TOKEN}}"
os.environ["HF_TOKEN"] = HF_TOKEN
CAND_REPO    = "Walidrbh27/khidmeti-stt-ctc-cand"
SERVED_MODEL = "Walidrbh27/khidmeti-stt"      # le whisper réellement servi
GATE_CONST   = 0.6300                          # la constante à revérifier
CTC_WER      = 0.6389                          # mesuré au run LM v4
THREADS      = 4                               # = STT_THREADS en prod

# Réglages EXACTS de docker/ai-stt/server.py — c'est ce que l'utilisateur reçoit.
LANGUAGE = "ar"
BEAM     = 1
VAD      = True
PROMPT   = ("خاصني بلومبي كهربائي سباك نجار حداد سودور بناي ماصون صباغ حلاق كوافور "
            "خياط طباخة جباص بلاكو ميكانيسيان كليماتيزور فريجيدار ماشينة تنظيف "
            "ديمناجمون تصليح نحوس على واحد يجيني للدار")

# ---- normalisation arabe (IDENTIQUE v3/v6/v7/v8/v9/v10 — comparabilité) -------
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

def left(): return DEADLINE_S - (time.time() - T0)

assert norm("الأستاذَة  [bruit] ى!") == "الاستاذه ي"

# ══ Levenshtein au niveau MOT, écrit ici : le bootstrap a besoin des éditions
# ══ PAR ÉNONCÉ, ce que jiwer.wer (agrégé) ne rend pas. Validé contre jiwer.
def edits(r, h):
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1]

assert edits("ا ب ت".split(), "ا ب ت".split()) == 0
assert edits("ا ب ت".split(), "ا ب".split()) == 1            # suppression
assert edits("ا ب".split(), "ا ب ت".split()) == 1            # insertion
assert edits("ا ب ت".split(), "ا ز ت".split()) == 1          # substitution
assert edits("ا ب ت".split(), "".split()) == 3

# ══ 1. données : les MÊMES 831 clips, même ordre ═══════════════════════════════
print("Loading Casablanca:Algeria …", flush=True)
ds = load_dataset("UBC-NLP/Casablanca", "Algeria")
ds = ds.cast_column("audio", Audio(sampling_rate=16000))
test = []
for ex in ds["test"]:
    t, d = clean(ex["transcription"]), float(ex["duration"])
    if 0.5 < d < 29.5 and len(t) > 1:
        test.append({"text": t, "dur": d,
                     "array": np.asarray(ex["audio"]["array"], dtype=np.float32)})
assert len(test) == 831, f"test différent des runs précédents : {len(test)}"
h_audio = sum(i["dur"] for i in test) / 3600.0
print(f"test={len(test)} ({h_audio:.2f} h d'audio)", flush=True)

# ══ 2. hypothèses CTC déjà mesurées + VÉRIFICATION d'alignement ═══════════════
# Un décalage d'un énoncé rendrait le test apparié absurde en silence : on exige
# que les références sauvées correspondent, une par une.
hj = json.load(open(hf_hub_download(CAND_REPO, "hyps.json", token=HF_TOKEN),
                    encoding="utf-8"))
ctc_hyps, ctc_refs = hj["hyps"], hj["refs"]
assert len(ctc_hyps) == len(test), f"{len(ctc_hyps)} hyps vs {len(test)} clips"
mism = [i for i in range(len(test)) if (norm(test[i]["text"]) or "-") != ctc_refs[i]]
assert not mism, f"alignement rompu sur {len(mism)} énoncés, ex. {mism[:5]}"
print(f"CTC hyps alignées ({hj['config']}) WER rapporté={hj['wer']}", flush=True)

R = [(norm(i["text"]) or "-").split() for i in test]
N_ref = [len(r) for r in R]
e_ctc = [edits(R[i], ctc_hyps[i].split()) for i in range(len(R))]
wer_ctc = sum(e_ctc) / sum(N_ref)
print(f"CTC WER revérifié = {wer_ctc:.4f} (sauvé {hj['wer']})", flush=True)
assert abs(wer_ctc - hj["wer"]) < 5e-4, "mon Levenshtein ≠ jiwer du run LM"

# ══ 3. whisper SERVI, réglages de production, sur les mêmes clips ══════════════
from faster_whisper import WhisperModel
print(f"chargement {SERVED_MODEL} int8 {THREADS} threads …", flush=True)
wm = WhisperModel(SERVED_MODEL, device="cpu", compute_type="int8", cpu_threads=THREADS)
print(f"réglages PRODUCTION : language={LANGUAGE} beam={BEAM} vad={VAD} "
      f"prompt={'oui' if PROMPT else 'non'}", flush=True)

wh_hyps, t_wh, done = [], 0.0, 0
for k, it in enumerate(test):
    t1 = time.time()
    try:
        segs, _ = wm.transcribe(it["array"], language=LANGUAGE, beam_size=BEAM,
                                vad_filter=VAD, initial_prompt=PROMPT,
                                condition_on_previous_text=False)
        txt = " ".join(s.text.strip() for s in segs).strip()
    except Exception as e:                       # un clip pourri ne tue pas 4 h
        print(f"  WARN_CLIP_{k}: {type(e).__name__}: {str(e)[:80]}", flush=True)
        txt = ""
    t_wh += time.time() - t1
    wh_hyps.append(txt)
    done = k + 1
    if done % 100 == 0:
        part = sum(edits(R[i], (norm(wh_hyps[i]) or "-").split())
                   for i in range(done)) / max(1, sum(N_ref[:done]))
        print(f"  [WHISPER] {done}/{len(test)}  WER_partiel={part:.4f}  "
              f"{t_wh/max(1e-9,sum(i['dur'] for i in test[:done])):.2f}×RT  "
              f"{left()/3600:.1f} h restantes", flush=True)
    if left() < 900:                             # 15 min : il faut publier
        print(f"DEADLINE_STOP après {done} clips", flush=True)
        break

H_wh = [(norm(h) or "-").split() for h in wh_hyps]
n = done                                          # apparié = uniquement les faits
e_wh = [edits(R[i], H_wh[i]) for i in range(n)]
wer_wh = sum(e_wh) / sum(N_ref[:n])
wh_rt = t_wh / max(1e-9, sum(i["dur"] for i in test[:n]))
# CER : jiwer sur les mêmes n énoncés, pour rester comparable aux runs précédents
cer_wh = jiwer.cer([" ".join(r) for r in R[:n]],
                   [" ".join(h) or "-" for h in H_wh[:n]])
cer_ctc = jiwer.cer([" ".join(r) for r in R[:n]],
                    [(ctc_hyps[i] or "-") for i in range(n)])
print(f"\nWHISPER_WER={wer_wh:.4f} CER={cer_wh:.4f} {wh_rt:.3f}×RT sur {n} clips",
      flush=True)
print(f"CTC_WER={sum(e_ctc[:n])/sum(N_ref[:n]):.4f} CER={cer_ctc:.4f} "
      f"(0,300×RT mesuré au run LM)", flush=True)
if abs(wer_wh - GATE_CONST) > 0.02:
    print(f"WARN_GATE_CONST_STALE : whisper mesuré {wer_wh:.4f} contre la constante "
          f"{GATE_CONST:.4f} du gate — les réglages de PRODUCTION (vad+prompt) ne "
          f"sont pas ceux qui ont produit {GATE_CONST:.4f}", flush=True)

# ══ 4. bootstrap APPARIÉ sur la différence ════════════════════════════════════
# Unité de rééchantillonnage = l'énoncé, et les DEUX systèmes suivent le même
# tirage : la difficulté de l'énoncé se compense, la variance chute.
e_c = e_ctc[:n]
wer_ctc_n = sum(e_c) / sum(N_ref[:n])        # CTC sur les n énoncés APPARIÉS
d_obs = wer_ctc_n - wer_wh
rnd = random.Random(20260819)
B, diffs, w_c, w_w = 4000, [], [], []
for _ in range(B):
    idx = [rnd.randrange(n) for _ in range(n)]
    den = sum(N_ref[i] for i in idx) or 1     # MÊME dénominateur pour les deux
    a = sum(e_c[i] for i in idx) / den
    b = sum(e_wh[i] for i in idx) / den
    w_c.append(a); w_w.append(b); diffs.append(a - b)
diffs.sort(); w_c.sort(); w_w.sort()
lo, hi = diffs[int(0.025 * B)], diffs[int(0.975 * B)]
sd = (sum((x - d_obs) ** 2 for x in diffs) / (B - 1)) ** 0.5
p_ctc_better = sum(1 for x in diffs if x < 0) / B

print("\n┌─ bootstrap APPARIÉ (4000 tirages, unité = énoncé) ─")
print(f"│ n énoncés appariés ............. {n}")
print(f"│ WER CTC ........................ {wer_ctc_n:.4f}  "
      f"IC95 [{w_c[int(0.025*B)]:.4f} ; {w_c[int(0.975*B)]:.4f}]")
print(f"│ WER whisper (prod) ............. {wer_wh:.4f}  "
      f"IC95 [{w_w[int(0.025*B)]:.4f} ; {w_w[int(0.975*B)]:.4f}]")
print(f"│ DIFFÉRENCE (CTC − whisper) ..... {d_obs:+.4f}  "
      f"IC95 [{lo:+.4f} ; {hi:+.4f}]  σ={sd:.4f}")
tie = lo <= 0.0 <= hi
print(f"│ zéro dans l'IC de la diff ? .... {'OUI → ÉGALITÉ statistique' if tie else 'NON → écart RÉEL'}")
print(f"│ P(CTC meilleur que whisper) .... {p_ctc_better*100:.1f} %")
print(f"│ CER CTC / whisper .............. {cer_ctc:.4f} / {cer_wh:.4f} "
      f"({'CTC' if cer_ctc < cer_wh else 'whisper'} devant)")
print(f"│ vitesse ........................ CTC 0,300×RT contre whisper "
      f"{wh_rt:.3f}×RT = ×{wh_rt/0.300:.1f}")
print(f"│ attente réelle, clip de 10 s ... CTC {10*0.300:.1f} s contre whisper "
      f"{10*wh_rt:.0f} s")
print("└─", flush=True)

# Par énoncé : qui gagne, et de combien. Un WER agrégé cache l'essentiel.
win = sum(1 for i in range(n) if e_c[i] < e_wh[i])
loss = sum(1 for i in range(n) if e_c[i] > e_wh[i])
draw = n - win - loss
print(f"par énoncé : CTC gagne {win} · perd {loss} · égalité {draw} "
      f"({win/n*100:.1f} % / {loss/n*100:.1f} % / {draw/n*100:.1f} %)", flush=True)
perfect_c = sum(1 for i in range(n) if e_c[i] == 0)
perfect_w = sum(1 for i in range(n) if e_wh[i] == 0)
print(f"énoncés parfaits : CTC {perfect_c} · whisper {perfect_w}", flush=True)

print("\n── 5 énoncés où whisper gagne le plus (pour lecture humaine) ──", flush=True)
for i in sorted(range(n), key=lambda i: e_c[i] - e_wh[i], reverse=True)[:5]:
    print(f"  RÉF : {' '.join(R[i])[:90]}")
    print(f"  CTC : {ctc_hyps[i][:90]}")
    print(f"  WHI : {' '.join(H_wh[i])[:90]}")

# ══ 5. artefacts : whisper_hyps.json → tout futur test apparié est gratuit ═════
os.makedirs("out", exist_ok=True)
json.dump({"model": SERVED_MODEL, "n": n,
           "settings": {"language": LANGUAGE, "beam": BEAM, "vad_filter": VAD,
                        "initial_prompt": bool(PROMPT), "threads": THREADS,
                        "compute_type": "int8"},
           "refs": [" ".join(r) for r in R[:n]], "hyps": wh_hyps[:n],
           "wer": round(wer_wh, 4), "cer": round(cer_wh, 4),
           "xrt_cpu": round(wh_rt, 4),
           "paired_vs_ctc": {"ctc_wer": round(sum(e_c)/sum(N_ref[:n]), 4),
                             "diff": round(d_obs, 4),
                             "diff_ci95": [round(lo, 4), round(hi, 4)],
                             "diff_sd": round(sd, 4), "tie": bool(tie),
                             "p_ctc_better": round(p_ctc_better, 4),
                             "utt_win": win, "utt_loss": loss, "utt_draw": draw},
           "gate_const_checked": {"constant": GATE_CONST,
                                  "measured_production_settings": round(wer_wh, 4),
                                  "stale": bool(abs(wer_wh - GATE_CONST) > 0.02)}},
          open("out/whisper_hyps.json", "w"), ensure_ascii=False)
HfApi(token=HF_TOKEN).upload_file(path_or_fileobj="out/whisper_hyps.json",
                                  path_in_repo="whisper_hyps.json",
                                  repo_id=CAND_REPO)
print(f"\nUPLOAD_DONE {CAND_REPO}/whisper_hyps.json", flush=True)
print(f"PAIRED_TIE={'YES' if tie else 'NO'} diff={d_obs:+.4f} "
      f"[{lo:+.4f};{hi:+.4f}] en {(time.time()-T0)/3600:.1f} h", flush=True)
