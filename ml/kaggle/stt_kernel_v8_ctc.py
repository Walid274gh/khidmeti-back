#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
# ml/kaggle/stt_kernel_v8_ctc.py — v8 : on change de FAMILLE, pas de taille.
#
# Constat (mesuré, run v7, MÊME test 831 clips Casablanca-Algeria, MÊME moteur) :
#   whisper-medium+LoRA servi ....... WER 0,6300  CER 0,2373
#   whisper-large-v3 nu ............. WER 0,8055  CER 0,3708   ← le socle whisper
#                                     n'apporte RIEN en darija : tout le gain
#                                     vient du LoRA sur 2 300 clips.
# Donc : payer l'encodeur whisper (fenêtre FIXE de 30 s, ~10× de calcul jeté sur
# une requête de 3-5 s) pour un socle qui ne sait pas la langue est absurde.
#
# v8 = wav2vec2 CTC, déjà pré-entraîné darija :
#   • boumehdi/wav2vec2-large-xlsr-moroccan-darija (Wav2Vec2ForCTC, 315 M,
#     transformers natif, vocab caractères arabe) — entraîné sur DVoice Darija.
#   • CTC = UNE passe avant, coût ∝ durée réelle (pas de padding 30 s), pas de
#     décodage auto-régressif, pas d'hallucination possible.
#   • Servi en onnxruntime int8/fp32 CPU (même moteur que ai-nlu/ai-vision).
#
# Entraînement 2 étages (curriculum) :
#   A. darija générique : DVoice v2.0 (cc-by-4.0, le corpus du socle) +
#      Casablanca:Morocco → 1 epoch. (adiren7 écarté : ses end_time sont des
#      offsets absolus, pas des durées — filtre de durée inutilisable.)
#   B. cible algérienne : Casablanca:Algeria ×2 → N epochs, val WER par epoch,
#      meilleur checkpoint gardé. Le val/test restent 100 % algériens.
#
# GATE = WER(candidat, moteur de service) < 0,6300 sur les MÊMES 831 clips.
#   échec → AUCUN remplacement : le repo whisper servi n'est pas touché, les
#   poids CTC restent sur le repo candidat pour inspection.
#
# v9 (même fichier, run 2) — CORRECTION DE MÉTHODE, pas de changement de modèle.
# Le run v8 a échoué le gate (0,7019 vs 0,6300) mais était SOUS-ENTRAÎNÉ : val WER
# encore en baisse monotone à la dernière époque (0,8061 → 0,7333 → 0,7121 →
# 0,7039 → 0,6957) et 0,4 h consommées sur 7,6 h de budget. Le plafond d'époques
# était donc la vraie limite mesurée, pas le modèle. v9 :
#   • étage B jusqu'à 16 époques avec arrêt sur plateau (patience 3),
#   • int8 quantifié sur les MatMul seulement (en v8 l'int8 était à la fois plus
#     mauvais 0,7311 et PLUS LENT 0,469×RT que le fp32 0,268×RT : la
#     quantification des convolutions du feature encoder coûte plus qu'elle ne
#     rapporte),
#   • latence CPU du whisper SERVI mesurée dans le même kernel : sans elle, la
#     question «plus léger/plus rapide ?» reste sans chiffre comparable.
#
# Leçons câblées (v6 : carte modèle crue sans vérif ; v7 : 6 h de poids perdus) :
#   1. Vocabulaire vérifié AVANT l'entraînement (aller-retour tokenizer + liste
#      des caractères hors-vocab, extension du lm_head si besoin).
#   2. Export ONNX + smoke WER 3 clips juste après l'entraînement, puis UPLOAD
#      DU CANDIDAT AVANT les evals longues.
#   3. Décodage de service (argmax + collapse + blank) écrit ICI et rejoué à
#      l'identique dans docker/ai-stt/server.py via ctc_vocab.json.
# ══════════════════════════════════════════════════════════════════════════════
import gc, json, os, random, re, subprocess, sys, time

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
REPO       = "Walidrbh27/khidmeti-stt-ctc"        # neuf : le repo whisper reste servi
CAND_REPO  = "Walidrbh27/khidmeti-stt-ctc-cand"   # filet : poids sauvés avant les evals

SERVED_WER, SERVED_CER = 0.6300, 0.2373   # whisper medium+LoRA int8, même test 831
ZS_LARGE_V3_WER        = 0.8055           # whisper-large-v3 nu, même test

EP_A, EP_B      = 1, 16                   # étage A (darija générique), B (algérien)
PATIENCE        = 3                       # v8 s'est arrêté sur le plafond, pas sur un plateau
SERVED_MODEL    = "Walidrbh27/khidmeti-stt"  # whisper servi : latence CPU de référence
LR_A, LR_B      = 1e-4, 3e-5
BATCH, ACCUM    = 2, 8
MAX_DUR_TRAIN   = 20.0                    # >20 s : hors entraînement (RAM T4), gardé au test
AUX_DVOICE      = 2500                    # plafond (temps mur, pas dogme)
DEV = "cuda"

# ---- normalisation arabe pour WER (IDENTIQUE v3/v6/v7 — comparabilité stricte) ----
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
#   "array" (déjà en RAM)  |  "path" (wav sur disque)  |  ("ds","i") (Arrow mmap)
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
random.shuffle(tr_full)
n_val = max(1, len(tr_full) // 10)
val, train_dz = tr_full[:n_val], tr_full[n_val:]
assert len(test) == 831, f"test différent des runs v3/v7 : {len(test)}"   # comparabilité
print(f"dz train={len(train_dz)} val={len(val)} test={len(test)}", flush=True)

# --- aux étage A : DVoice v2.0 Darija (le corpus du socle, cc-by-4.0) ---------
# datasets 3.x ne charge plus les scripts de dataset (darija20.py) : on lit
# metadata.csv et on tire les wav directement, seul le sous-ensemble retenu.
aux = []
try:
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import hf_hub_download
    import csv
    md = hf_hub_download("BrunoHays/DVOICEv2.0-Darija", "metadata.csv",
                         repo_type="dataset", token=HF_TOKEN)
    rows, seen = [], set()
    with open(md, encoding="utf-8") as f:
        for r in csv.DictReader(f):     # colonnes vérifiées : file_name,words,duration,audio_id
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
print(f"étage A={len(aux)} clips  étage B={len(train_b)} clips  "
      f"(DZ >{MAX_DUR_TRAIN:.0f}s écartés: {len(train_dz)-len(train_b)//2})", flush=True)
assert len(train_b) > 500, "trop peu de données algériennes — abandon"

# ══ 2. vocabulaire : vérifié AVANT d'entraîner (leçon v6) ═════════════════════
processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL)
tok, fx = processor.tokenizer, processor.feature_extractor
BLANK = tok.pad_token_id
vocab = tok.get_vocab()
chars = {c for it in (train_b + aux) for c in it["text"]} - {" "}
missing = sorted(chars - set(vocab))
print(f"vocab={len(vocab)} blank={BLANK} caractères corpus={len(chars)} "
      f"hors-vocab={missing}", flush=True)
if missing:
    tok.add_tokens(missing)          # ids ajoutés à la fin → lm_head élargi plus bas
    vocab = tok.get_vocab()
    print(f"vocab étendu → {len(vocab)}", flush=True)

# aller-retour tokenizer : group_tokens=False, sinon le décodage CTC écrase les
# lettres doublées («الله») et l'assert échouerait pour la mauvaise raison.
for it in random.sample(train_b + aux, min(200, len(train_b + aux))):
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

def keep_if_best(w, c, tag):
    if w < best["wer"]:
        best.update(wer=w, cer=c, tag=tag,
                    state={k: v.detach().cpu().clone()
                           for k, v in model.state_dict().items()})
        print(f"NEW_BEST={tag} val_WER={w:.4f}", flush=True)

def stage(items, epochs, lr, tag):
    global skipped
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=lr, weight_decay=0.01)
    steps = max(1, len(items) // (BATCH * ACCUM)) * epochs
    sched = get_linear_schedule_with_warmup(opt, max(1, int(steps * 0.1)), steps)
    scaler = torch.amp.GradScaler("cuda")
    order = list(range(len(items)))
    bad = 0
    for ep in range(epochs):
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

print(f"=== socle nu (zero-shot darija marocain) — {left()/3600:.1f} h ===", flush=True)
zs_val = torch_eval(val, "ZS_BASE_val")
if aux:
    print(f"=== étage A : darija générique {len(aux)} clips ===", flush=True)
    stopped = stage(aux, EP_A, LR_A, "A")
if not stopped:
    print(f"=== étage B : algérien {len(train_b)} clips ===", flush=True)
    stopped = stage(train_b, EP_B, LR_B, "B")
assert best["state"] is not None, "aucun checkpoint validé"
model.load_state_dict(best["state"])
print(f"BEST={best['tag']} val_WER={best['wer']:.4f} (socle nu {zs_val[0]:.4f})", flush=True)
if best["wer"] > zs_val[0]:      # le fine-tune a dégradé le socle : à savoir AVANT le gate
    print(f"WARN_FT_WORSE_THAN_BASE: {best['wer']:.4f} > {zs_val[0]:.4f}", flush=True)

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
print(f"│ wav2vec2 darija nu (val) .............. WER {zs_val[0]:.4f}")
print(f"│ wav2vec2 darija v8 (torch, test) ...... WER {full_wer:.4f} CER {full_cer:.4f}")
print(f"│ wav2vec2 darija v8 (servi, estimé) .... WER {wer_served_est:.4f}")
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
    "version": "v9-ctc", "trained": "2026-08-17", "seed": SEED,
    "arch": "Wav2Vec2ForCTC (encodeur seul, décodage CTC glouton)",
    "base_model": BASE_MODEL,
    "why_this_base": "déjà pré-entraîné sur la darija (DVoice) contrairement à whisper "
                     "(large-v3 nu = 0,8055 sur ce test : le socle whisper n'apporte "
                     "rien en darija, tout venait du LoRA) ; CTC = une passe, coût "
                     "proportionnel à la durée réelle, pas de fenêtre 30 s à remplir",
    "method": f"2 étages : A darija générique {EP_A} ép lr {LR_A}, "
              f"B algérien ×2 jusqu'à {EP_B} ép lr {LR_B} (arrêt sur plateau, "
              f"patience {PATIENCE}) ; feature encoder gelé, "
              "spec-augment, fp16, clip 1.0",
    "hparams": {"batch": BATCH, "accum": ACCUM, "max_dur_train": MAX_DUR_TRAIN,
                "best": best["tag"], "skipped_nonfinite_steps": skipped,
                "deadline_stop": stopped or None, "vocab_added": missing},
    "datasets": {"train_dz": "UBC-NLP/Casablanca:Algeria validation ×2 (CC BY-NC-ND 4.0)",
                 "aux": "BrunoHays/DVOICEv2.0-Darija (CC BY 4.0) + Casablanca:Morocco",
                 "test": "UBC-NLP/Casablanca:Algeria test, 831 clips utilisables"},
    "license_note": "poids intérimaires non-commerciaux (Casablanca CC BY-NC-ND) — "
                    "remplacer par les données flywheel avant commercialisation",
    "rows": {"train_dz": len(train_dz), "train_b": len(train_b),
             "dvoice": sum(1 for a in aux if "path" in a),
             "casablanca_ma": sum(1 for a in aux if "ds" in a),
             "val": len(val), "test": len(test)},
    "wer_norm": {"served_whisper_v3": SERVED_WER, "zeroshot_whisper_large_v3": ZS_LARGE_V3_WER,
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

api.create_repo(REPO, private=True, exist_ok=True)
api.upload_folder(folder_path="out", repo_id=REPO)
print("HF_UPLOAD_DONE=" + REPO)
print("\n── à mettre dans .env.cloud ET .env.local (jamais le .env généré) ──")
print("STT_ENGINE=ctc")
print(f"STT_MODEL={REPO}")
print(f"STT_MODEL_FILE={serve_file}")
print("# retour arrière : STT_ENGINE=whisper (le repo whisper reste intact)")
print(f"V8 COMPLETE: WER {wer_served_est:.4f} (servi {SERVED_WER:.4f}) "
      f"en {(time.time()-T0)/3600:.1f} h")





