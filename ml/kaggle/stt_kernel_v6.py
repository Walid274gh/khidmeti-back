#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
# ⛔ NE PAS LANCER — DÉPASSÉ (15 août 2026). Remplacé par stt_kernel_v7b.py.
#
# BASE_MODEL ci-dessous (anaszil/whisper-large-v3-turbo-darija) est un
# adaptateur PEFT/LoRA : WhisperForConditionalGeneration.from_pretrained le
# refuse, et même converti faster-whisper ne le sert pas. Ce kernel n'a jamais
# tourné et échouerait au chargement.
#
# Mesuré depuis (831 clips Casablanca-Algeria, moteur de prod int8) :
#   v3 servi medium+LoRA 0,6300 · openai/whisper-large-v3 nu 0,8055
#   large-v3 + LoRA r32 : meilleur val 0,5738
# Le gain vient du fine-tuning algérien, pas du socle. Gate WER<0,50 irréaliste.
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# ml/kaggle/stt_kernel_v6.py — STT v6 with enhanced training for 15k+ clips
#
# Improvements over v3:
#   - Larger LoRA rank (r=64) for 10x data
#   - Gradient accumulation ×8 for stability
#   - Extended epochs (8) with early stopping
#   - CER metric tracking (Arabic script sensitivity)
#   - Per-duration-bin stratified eval
#   - Spec augment tuning for real-world noise
#   - Target WER < 0.50 (vs v3 0.6303)
# ══════════════════════════════════════════════════════════════════════════════
import gc, json, os, random, re, subprocess, sys

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

SEED = 20260810
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

HF_TOKEN = "{{HF_TOKEN}}"
os.environ["HF_TOKEN"] = HF_TOKEN
REPO   = "Walidrbh27/khidmeti-stt"
BASE_MODEL = "anaszil/whisper-large-v3-turbo-darija"  # pre-trained on Darija!
EPOCHS, LR, BATCH, ACCUM = 6, 5e-5, 2, 16  # v3-turbo larger: smaller batch, lower LR
GEN_BS, MAX_LABEL = 4, 448
DEV = "cuda"

# ---- normalisation arabe pour WER (identique v3 — comparabilité) ----
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

# ---- data : Algérie (train/val/test) + Maroc (aux train uniquement) ----
def usable(ex):
    t = re.sub(r"<[^>]*>|\[[^\]]*\]", "", ex["transcription"]).strip()
    return 0.5 < ex["duration"] < 29.5 and len(t) > 1

print("Loading Casablanca dataset...")
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

# DZ sur-échantillonné ×2, val/test 100% algériens
train = train_dz * 2 + aux_ma
print(f"train_dz={len(train_dz)} aux_ma={len(aux_ma)} train={len(train)} "
      f"val={len(val)} test={len(test)}")

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
    model.eval()
    hyps = []
    for i in range(0, len(items), GEN_BS):
        feats = collate(items[i:i+GEN_BS], with_labels=False).to(DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model.generate(input_features=feats, language="ar",
                                 task="transcribe", max_new_tokens=225, num_beams=1)
        hyps += tok.batch_decode(out, skip_special_tokens=True)
        if i == 0:
            print(f"[{tag}] hyp0: {hyps[0][:120]}")
    return hyps

def fw_eval(model_dir, items, tag):
    """Eval avec le MOTEUR DE PROD (faster-whisper int8) — comparaison à iso-format."""
    from faster_whisper import WhisperModel
    fw = WhisperModel(model_dir, device="cuda", compute_type="int8_float16")
    hyps = []
    for k, ex in enumerate(items):
        segs, _ = fw.transcribe(ex["audio"]["array"].astype(np.float32),
                                language="ar", beam_size=1)
        hyps.append(" ".join(s.text.strip() for s in segs))
        if k == 0:
            print(f"[{tag}] hyp0: {hyps[0][:120]}")
    del fw; gc.collect(); torch.cuda.empty_cache()
    w, c = wer_cer([x["transcription"] for x in items], hyps)
    print(f"{tag}_WER={w:.4f} {tag}_CER={c:.4f}")
    return w, c

# ---- baseline = le modèle SERVI aujourd'hui (khidmeti-stt v3), int8, full test ----
from huggingface_hub import snapshot_download, HfApi
print("Evaluating baseline (served v3)...")
served_dir = snapshot_download(REPO, token=HF_TOKEN)
served_wer, served_cer = fw_eval(served_dir, test, "BASELINE_SERVED_V3")

# ---- medium + LoRA r64 (doubled for 10x data) ----
from peft import LoraConfig, get_peft_model, \
    get_peft_model_state_dict, set_peft_model_state_dict

print("Building model with LoRA r=64...")
base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
base.generation_config.forced_decoder_ids = None
base.generation_config.use_cache = True
cfg = base.config
cfg.use_cache = False
cfg.apply_spec_augment = True
cfg.mask_time_prob = 0.08      # increased for robustness
cfg.mask_feature_prob = 0.03   # light freq masking
base.model.encoder.conv1.register_forward_hook(
    lambda m, i, o: o.requires_grad_(True))
base.gradient_checkpointing_enable()

model = get_peft_model(base, LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])).to(DEV)
model.print_trainable_parameters()

opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
steps = (len(train) // (BATCH * ACCUM) + 1) * EPOCHS
sched = get_linear_schedule_with_warmup(opt, int(steps * 0.1), steps)
scaler = torch.amp.GradScaler("cuda")

best_wer, best_cer, best_adapter = float("inf"), float("inf"), None
order = list(range(len(train)))

print(f"Training for {EPOCHS} epochs...")
for ep in range(EPOCHS):
    model.train(); random.shuffle(order); tot, nb = 0.0, 0
    opt.zero_grad()
    for j in range(0, len(order), BATCH):
        feats, labs = collate([train[k] for k in order[j:j+BATCH]])
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_features=feats.to(DEV), labels=labs.to(DEV)).loss / ACCUM
        scaler.scale(loss).backward()
        tot += loss.item() * ACCUM; nb += 1
        if (j // BATCH) % ACCUM == ACCUM - 1:
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()

    # Validation
    cfg.apply_spec_augment = False
    val_hyps = transcribe_all(model, val, f"val_ep{ep+1}")
    vw, vc = wer_cer([x["transcription"] for x in val], val_hyps)
    cfg.apply_spec_augment = True

    print(f"epoch {ep+1} loss={tot/max(nb,1):.4f} val_WER={vw:.4f} val_CER={vc:.4f}")

    if vw < best_wer:
        best_wer = vw
        best_cer = vc
        best_adapter = {k: v.detach().cpu().clone()
                        for k, v in get_peft_model_state_dict(model).items()}

cfg.apply_spec_augment = False
set_peft_model_state_dict(model, best_adapter)

# ---- merge LoRA → poids pleins, export CT2 int8 ----
print("Merging LoRA and exporting to CTranslate2 int8...")
merged = model.merge_and_unload().cpu()
cfg.use_cache = True
merged.save_pretrained("ft_hf"); processor.save_pretrained("ft_hf")
del model, merged, base; gc.collect(); torch.cuda.empty_cache()

from ctranslate2.converters import TransformersConverter
cfiles = [f for f in ("tokenizer.json", "preprocessor_config.json")
          if os.path.exists(f"ft_hf/{f}")]
TransformersConverter("ft_hf", copy_files=cfiles) \
    .convert("ct2_out", quantization="int8", force=True)

# ---- eval nouveau modèle int8, full test, même moteur ----
print("Evaluating new model int8...")
new_wer, new_cer = fw_eval("ct2_out", test, "FT_MEDIUM_V6_INT8")

# ---- gate : doit battre le modèle SERVI ET WER < 0.50 ----
gate_pass = new_wer < served_wer and new_wer < 0.50
print(f"GATE_BEATS_SERVED={'YES' if new_wer < served_wer else 'NO'} "
      f"(new={new_wer:.4f} vs served={served_wer:.4f})")
print(f"GATE_WER_50={'YES' if new_wer < 0.50 else 'NO'} (target < 0.50)")
print(f"GATE_PASS={'YES' if gate_pass else 'NO'}")

if not gate_pass:
    print("GATE_FAILED_NO_UPLOAD — v6 ne passe pas les deux gates")
    sys.exit(0)

# ---- duration-stratified analysis ----
print("\nDuration-stratified analysis:")
bins = [(0.5, 5), (5, 10), (10, 20), (20, 30)]
for low, high in bins:
    subset = [ex for ex in test if low < ex["duration"] <= high]
    if subset:
        hyps = []
        fw = WhisperModel("ct2_out", device="cuda", compute_type="int8_float16")
        for ex in subset:
            segs, _ = fw.transcribe(ex["audio"]["array"].astype(np.float32),
                                   language="ar", beam_size=1)
            hyps.append(" ".join(s.text.strip() for s in segs))
        w, c = wer_cer([x["transcription"] for x in subset], hyps)
        print(f"  {low}-{high}s ({len(subset)} clips): WER={w:.4f} CER={c:.4f}")
        del fw; gc.collect(); torch.cuda.empty_cache()

# ---- package + upload ----
json.dump({"base_model": BASE_MODEL, "method": "LoRA r64 a128 qkvo+fc merged",
           "dataset": "UBC-NLP/Casablanca:Algeria x2 + Morocco aux (CC BY-NC-ND 4.0)",
           "license_note": "poids intérimaires non-commerciaux — remplacer via flywheel P6",
           "version": "v6", "trained": "2026-08-10", "seed": SEED,
           "improvement": "BASE whisper-large-v3-turbo-darija instead of raw medium",
           "train_dz": len(train_dz), "aux_ma": len(aux_ma),
           "val_rows": len(val), "test_rows": len(test),
           "wer_norm": {"served_v3_int8": served_wer, "ft_medium_v6_int8": new_wer,
                        "best_val_wer": best_wer, "best_val_cer": best_cer},
           "cer_norm": {"served_v3_int8": served_cer, "ft_medium_v6_int8": new_cer},
           "gate": "WER < served AND WER < 0.50",
           "serve": "faster-whisper compute_type=int8 language=ar"},
          open("ct2_out/meta.json", "w"), indent=2)

print("\nUploading to HuggingFace...")
api = HfApi(token=HF_TOKEN)
api.upload_folder(folder_path="ct2_out", repo_id=REPO)
print("HF_UPLOAD_DONE=" + REPO)
# Auto-restart Space to pick up new weights
try:
    from huggingface_hub import HfApi as _HfApi2
    _HfApi2(token=HF_TOKEN).restart_space(repo_id="Walidrbh27/khidmeti-ai", repo_type="space")
    print("SPACE_RESTARTED=Walidrbh27/khidmeti-ai")
except Exception as _e:
    print(f"SPACE_RESTART_SKIP={_e}")

print(f"\nv6 COMPLETE: WER {new_wer:.4f} CER {new_cer:.4f}")
