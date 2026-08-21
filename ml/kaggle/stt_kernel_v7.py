#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
# ml/kaggle/stt_kernel_v7.py — STT v7 : whisper-large-v3 + LoRA (Casablanca DZ)
#
# POURQUOI large-v3 et pas autre chose (choix « meilleur modèle », Aug 14 2026) :
#   contrainte dure = le conteneur ai-stt sert via faster-whisper/CTranslate2.
#   Seule l'architecture Whisper est convertible CT2. Cohere-Transcribe-Arabic
#   (Casablanca 49,71) et Audar (51,58) sont meilleurs mais non convertibles et
#   Audar a une licence restreinte. Dans la famille Whisper, classement tiers
#   (Open Universal Arabic ASR Leaderboard, colonne Casablanca, zero-shot) :
#     large-v3 71,81  <  large-v3-turbo 73,79  <  medium 75,44
#   => large-v3 est LE meilleur socle servable. Notre v3 servi part de medium.
#
# Trois mesures sur LE MÊME test set, LE MÊME moteur (faster-whisper int8) :
#   1. BASELINE_SERVED_V3   — ce qui tourne aujourd'hui (medium+LoRA, 0,6303)
#   2. ZEROSHOT_LARGE_V3    — le socle nu, sans entraînement  ← contrôle
#   3. FT_LARGE_V3_V7       — le socle + LoRA sur données algériennes
#   (2) isole ce qui vient du changement de socle, (3) ce qu'ajoute le fine-tune.
#   Sans (2) un bon résultat serait ininterprétable.
#
# Mesure bonus gratuite : eval AVEC initial_prompt métier (ce que le serveur
#   passe réellement en prod, server.py:70) — jamais mesuré jusqu'ici.
#
# Garde-fous T4 : deadline 7,6 h (limite Kaggle 9 h) → sort de la boucle et
#   exporte quand même ; skip des steps non-finis (fp16 sur 1,55 B).
# ══════════════════════════════════════════════════════════════════════════════
import gc, json, os, random, re, shutil, subprocess, sys, time

T0 = time.time()
DEADLINE_S = 7.6 * 3600          # ponytail: budget mur ; au-delà on exporte le meilleur checkpoint

os.environ["TOKENIZERS_PARALLELISM"] = "false"
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "datasets==3.6.0", "soundfile", "librosa", "jiwer",
                "ctranslate2", "faster-whisper", "peft"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

import numpy as np
import torch
import jiwer
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor, \
    get_linear_schedule_with_warmup

SEED = 20260814
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

HF_TOKEN = "{{HF_TOKEN}}"
os.environ["HF_TOKEN"] = HF_TOKEN
REPO       = "Walidrbh27/khidmeti-stt"
BASE_MODEL = "openai/whisper-large-v3"          # meilleur socle Whisper servable en CT2
EPOCHS, LR, BATCH, ACCUM = 5, 3e-5, 1, 16       # 1,55 B : batch 1 + accum 16 (eff. 16)
LORA_R, LORA_A = 32, 64                         # r=32 comme v3 — r=64 sur 1,55 B + ~2 k clips = surapprentissage
GEN_BS, MAX_LABEL = 2, 448
DEV = "cuda"

# Amorce métier IDENTIQUE à docker/ai-stt/server.py (_DEFAULT_PROMPT) — pour
# mesurer le WER tel que la prod le produit réellement, pas seulement à nu.
PROD_PROMPT = (
    "خاصني بلومبي كهربائي سباك نجار حداد سودور بناي ماصون صباغ حلاق كوافور "
    "خياط طباخة جباص بلاكو ميكانيسيان كليماتيزور فريجيدار ماشينة تنظيف "
    "ديمناجمون تصليح نحوس على واحد يجيني للدار"
)

# ---- normalisation arabe pour WER (identique v3/v6 — comparabilité stricte) ----
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

# self-check normalisation : casse en 2 s si la regex arabe est cassée, au lieu de
# rendre 5 h de WER incomparables avec v3.
assert norm("الأستاذَة  [bruit] ى!") == "الاستاذه ي", norm("الأستاذَة  [bruit] ى!")
assert wer_cer(["خاصني بلومبي"], ["خاصني بلومبي"])[0] == 0.0
assert wer_cer(["خاصني بلومبي"], ["خاصني كهربائي"])[0] == 0.5

# ---- data : Algérie (train/val/test) + Maroc (aux train uniquement) ----
def usable(ex):
    t = re.sub(r"<[^>]*>|\[[^\]]*\]", "", ex["transcription"]).strip()
    return 0.5 < ex["duration"] < 29.5 and len(t) > 1

print("Loading Casablanca:Algeria …", flush=True)
ds = load_dataset("UBC-NLP/Casablanca", "Algeria")
ds = ds.cast_column("audio", Audio(sampling_rate=16000))
tr_full = [ex for ex in ds["validation"] if usable(ex)]
test    = [ex for ex in ds["test"]       if usable(ex)]
random.shuffle(tr_full)
n_val = max(1, len(tr_full) // 10)
val, train_dz = tr_full[:n_val], tr_full[n_val:]

aux_ma = []
try:
    ma = load_dataset("UBC-NLP/Casablanca", "Morocco")
    ma = ma.cast_column("audio", Audio(sampling_rate=16000))
    aux_ma = [ex for ex in ma["validation"] if usable(ex)]
except Exception as e:
    print(f"WARN_NO_MOROCCO_AUX: {e}")

train = train_dz * 2 + aux_ma          # DZ sur-échantillonné ×2 ; val/test 100 % algériens
print(f"train_dz={len(train_dz)} aux_ma={len(aux_ma)} train={len(train)} "
      f"val={len(val)} test={len(test)}", flush=True)

processor = WhisperProcessor.from_pretrained(BASE_MODEL, language="Arabic", task="transcribe")
tok = processor.tokenizer
START = tok.convert_tokens_to_ids("<|startoftranscript|>")

def collate(batch, with_labels=True):
    feats = processor([b["audio"]["array"] for b in batch],
                      sampling_rate=16000, return_tensors="pt").input_features
    if not with_labels:
        return feats
    enc = tok([b["transcription"] for b in batch], return_tensors="pt",
              padding=True, truncation=True, max_length=MAX_LABEL)
    labs = enc.input_ids.masked_fill(enc.attention_mask.ne(1), -100)
    if (labs[:, 0] == START).all():
        labs = labs[:, 1:]
    return feats, labs

@torch.no_grad()
def transcribe_all(model, items, tag):
    model.eval(); hyps = []
    for i in range(0, len(items), GEN_BS):
        feats = collate(items[i:i+GEN_BS], with_labels=False).to(DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model.generate(input_features=feats, language="ar",
                                 task="transcribe", max_new_tokens=225, num_beams=1)
        hyps += tok.batch_decode(out, skip_special_tokens=True)
        if i == 0:
            print(f"[{tag}] hyp0: {hyps[0][:120]}", flush=True)
    return hyps

def fw_eval(model_dir, items, tag, prompt=None):
    """Eval avec le MOTEUR DE PROD (faster-whisper int8) — comparaison à iso-format."""
    from faster_whisper import WhisperModel
    fw = WhisperModel(model_dir, device="cuda", compute_type="int8_float16")
    hyps = []
    for k, ex in enumerate(items):
        segs, _ = fw.transcribe(ex["audio"]["array"].astype(np.float32),
                                language="ar", beam_size=1, initial_prompt=prompt)
        hyps.append(" ".join(s.text.strip() for s in segs))
        if k == 0:
            print(f"[{tag}] hyp0: {hyps[0][:120]}", flush=True)
    del fw; gc.collect(); torch.cuda.empty_cache()
    w, c = wer_cer([x["transcription"] for x in items], hyps)
    print(f"{tag}_WER={w:.4f} {tag}_CER={c:.4f}", flush=True)
    return w, c

def to_ct2(hf_dir, out_dir):
    from ctranslate2.converters import TransformersConverter
    cf = [f for f in ("tokenizer.json", "preprocessor_config.json")
          if os.path.exists(f"{hf_dir}/{f}")]
    TransformersConverter(hf_dir, copy_files=cf).convert(out_dir, quantization="int8", force=True)
    return out_dir

from huggingface_hub import snapshot_download, HfApi

# ---- (1) baseline = le modèle SERVI aujourd'hui (khidmeti-stt v3), int8, full test ----
print("=== (1) baseline servie v3 ===", flush=True)
served_wer, served_cer = fw_eval(snapshot_download(REPO, token=HF_TOKEN),
                                 test, "BASELINE_SERVED_V3")

# ---- (2) contrôle = large-v3 NU, même moteur, même test ----
# snapshot local d'abord : sans tokenizer.json dans le dossier CT2, faster-whisper
# retombe sur le tokenizer de whisper-tiny (51 865 tokens) alors que large-v3 en a
# 51 866 → décalage d'un token et décodage faussé. Ce piège coûterait tout le run.
print("=== (2) contrôle zero-shot large-v3 ===", flush=True)
base_snap = snapshot_download(BASE_MODEL, token=HF_TOKEN,
                              allow_patterns=["*.json", "*.txt", "model.safetensors"])
zs_wer, zs_cer = fw_eval(to_ct2(base_snap, "ct2_zs"), test, "ZEROSHOT_LARGE_V3")
shutil.rmtree("ct2_zs", ignore_errors=True)      # 20 Go de disque Kaggle : on nettoie
gc.collect(); torch.cuda.empty_cache()

# ---- (3) large-v3 + LoRA r32 ----
from peft import LoraConfig, get_peft_model, \
    get_peft_model_state_dict, set_peft_model_state_dict

print(f"=== (3) large-v3 + LoRA r{LORA_R} — {left()/3600:.1f} h restantes ===", flush=True)
base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
base.generation_config.forced_decoder_ids = None
base.generation_config.use_cache = True
cfg = base.config
cfg.use_cache = False
cfg.apply_spec_augment = True
cfg.mask_time_prob = 0.05
cfg.mask_feature_prob = 0.02
base.model.encoder.conv1.register_forward_hook(lambda m, i, o: o.requires_grad_(True))
base.gradient_checkpointing_enable()

model = get_peft_model(base, LoraConfig(
    r=LORA_R, lora_alpha=LORA_A, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])).to(DEV)
model.print_trainable_parameters()

opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
steps = (len(train) // (BATCH * ACCUM) + 1) * EPOCHS
sched = get_linear_schedule_with_warmup(opt, int(steps * 0.1), steps)
scaler = torch.amp.GradScaler("cuda")

best_wer, best_cer, best_ep, best_adapter = float("inf"), float("inf"), 0, None
order, skipped, stopped = list(range(len(train))), 0, ""

for ep in range(EPOCHS):
    model.train(); random.shuffle(order); tot, nb = 0.0, 0
    opt.zero_grad()
    for j in range(0, len(order), BATCH):
        feats, labs = collate([train[k] for k in order[j:j+BATCH]])
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_features=feats.to(DEV), labels=labs.to(DEV)).loss / ACCUM
        if not torch.isfinite(loss):      # fp16 sur 1,55 B : un step pourri ne tue pas le run
            skipped += 1; opt.zero_grad(); continue
        scaler.scale(loss).backward()
        tot += loss.item() * ACCUM; nb += 1
        if (j // BATCH) % ACCUM == ACCUM - 1:
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
        if left() < 3000:                 # 50 min réservées à l'export + les 2 evals
            stopped = f"deadline_epoch{ep+1}_step{j}"; break

    cfg.apply_spec_augment = False
    vw, vc = wer_cer([x["transcription"] for x in val],
                     transcribe_all(model, val, f"val_ep{ep+1}"))
    cfg.apply_spec_augment = True
    print(f"epoch {ep+1} loss={tot/max(nb,1):.4f} val_WER={vw:.4f} val_CER={vc:.4f} "
          f"skipped={skipped} left={left()/3600:.1f}h", flush=True)

    if vw < best_wer:
        best_wer, best_cer, best_ep = vw, vc, ep + 1
        best_adapter = {k: v.detach().cpu().clone()
                        for k, v in get_peft_model_state_dict(model).items()}
    if stopped:
        print(f"DEADLINE_STOP={stopped}", flush=True); break

assert best_adapter is not None, "aucun checkpoint validé"
cfg.apply_spec_augment = False
set_peft_model_state_dict(model, best_adapter)
print(f"best_epoch={best_ep} best_val_WER={best_wer:.4f}", flush=True)

# ---- merge LoRA → poids pleins fp16 → export CT2 int8 ----
merged = model.merge_and_unload().cpu().half()   # fp16 : 3,1 Go au lieu de 6,2 sur disque
cfg.use_cache = True
merged.save_pretrained("ft_hf"); processor.save_pretrained("ft_hf")
del model, merged, base; gc.collect(); torch.cuda.empty_cache()
to_ct2("ft_hf", "ct2_out")
shutil.rmtree("ft_hf", ignore_errors=True)

# ---- evals finales : à nu ET avec l'amorce métier que la prod passe vraiment ----
print(f"=== evals finales — {left()/3600:.1f} h restantes ===", flush=True)
new_wer,  new_cer  = fw_eval("ct2_out", test, "FT_LARGE_V3_V7")
newp_wer, newp_cer = fw_eval("ct2_out", test, "FT_LARGE_V3_V7_PROMPTED", prompt=PROD_PROMPT)
servedp_wer, servedp_cer = fw_eval(snapshot_download(REPO, token=HF_TOKEN),
                                   test, "BASELINE_SERVED_V3_PROMPTED", prompt=PROD_PROMPT)

print("\n┌─ récap (même test set, même moteur faster-whisper int8) ─")
print(f"│ servi v3 medium+LoRA   à nu {served_wer:.4f}   amorce {servedp_wer:.4f}")
print(f"│ large-v3 zero-shot     à nu {zs_wer:.4f}")
print(f"│ large-v3 +LoRA (v7)    à nu {new_wer:.4f}   amorce {newp_wer:.4f}")
print(f"│ apport du socle  : {served_wer - zs_wer:+.4f}")
print(f"│ apport du LoRA   : {zs_wer - new_wer:+.4f}")
print(f"│ apport de l'amorce : v3 {served_wer - servedp_wer:+.4f}  v7 {new_wer - newp_wer:+.4f}")
print("└─", flush=True)

best_new = min(new_wer, newp_wer)
best_old = min(served_wer, servedp_wer)
gate_pass = best_new < best_old
print(f"GATE_BEATS_SERVED={'YES' if gate_pass else 'NO'} "
      f"(v7 {best_new:.4f} vs servi {best_old:.4f})")
print(f"INFO_WER_55={'YES' if best_new < 0.55 else 'NO'}")
print(f"PROMPT_HELPS_V7={'YES' if newp_wer < new_wer else 'NO'} "
      f"→ STT_PROMPT={'on' if newp_wer < new_wer else 'off'}")
print(f"GATE_PASS={'YES' if gate_pass else 'NO'}", flush=True)

if not gate_pass:
    print("GATE_FAILED_NO_UPLOAD — v7 ne bat pas les poids servis ; on ne remplace rien.")
    sys.exit(0)

# ---- analyse par durée : où le modèle casse (clips courts = notre cas d'usage) ----
print("\nWER par tranche de durée :", flush=True)
strat, use_p = {}, newp_wer < new_wer
for low, high in [(0.5, 5), (5, 10), (10, 20), (20, 30)]:
    sub = [ex for ex in test if low < ex["duration"] <= high]
    if len(sub) >= 10:
        w, c = fw_eval("ct2_out", sub, f"BIN_{low}_{high}",
                       prompt=PROD_PROMPT if use_p else None)
        strat[f"{low}-{high}s"] = {"n": len(sub), "wer": w, "cer": c}

# ---- package + upload ----
json.dump({
    "version": "v7", "trained": "2026-08-14", "seed": SEED,
    "base_model": BASE_MODEL,
    "why_this_base": "meilleur socle Whisper sur Casablanca (71,81) et seule famille "
                     "convertible CTranslate2 ; Cohere 49,71 et Audar 51,58 sont "
                     "meilleurs mais non servables par faster-whisper (Audar : licence restreinte)",
    "method": f"LoRA r{LORA_R} a{LORA_A} qkvo+fc merged, fp16, spec-augment",
    "hparams": {"epochs": EPOCHS, "lr": LR, "batch": BATCH, "accum": ACCUM,
                "best_epoch": best_ep, "skipped_nonfinite_steps": skipped,
                "deadline_stop": stopped or None},
    "dataset": "UBC-NLP/Casablanca:Algeria x2 + Morocco aux (CC BY-NC-ND 4.0)",
    "license_note": "poids intérimaires non-commerciaux — remplacer via flywheel P6",
    "rows": {"train_dz": len(train_dz), "aux_ma": len(aux_ma),
             "train": len(train), "val": len(val), "test": len(test)},
    "wer_norm": {"served_v3": served_wer, "served_v3_prompted": servedp_wer,
                 "zeroshot_large_v3": zs_wer,
                 "ft_v7": new_wer, "ft_v7_prompted": newp_wer,
                 "best_val": best_wer},
    "cer_norm": {"served_v3": served_cer, "served_v3_prompted": servedp_cer,
                 "zeroshot_large_v3": zs_cer,
                 "ft_v7": new_cer, "ft_v7_prompted": newp_cer,
                 "best_val": best_cer},
    "wer_by_duration": strat,
    "gate": "min(WER à nu, WER avec amorce) < même minimum du modèle servi",
    "serve": "faster-whisper compute_type=int8 language=ar",
    "serve_prompt": "on" if use_p else "off",
    "warning_mem_limit": "large-v3 int8 ≈ 1,6 Go de poids : docker-compose ai-stt "
                         "mem_limit doit passer de 2g à 4g",
}, open("ct2_out/meta.json", "w"), indent=2, ensure_ascii=False)

HfApi(token=HF_TOKEN).upload_folder(folder_path="ct2_out", repo_id=REPO)
print("HF_UPLOAD_DONE=" + REPO)
print(f"V7 COMPLETE: WER {best_new:.4f} (servi {best_old:.4f}) "
      f"en {(time.time()-T0)/3600:.1f} h")
