# Khidmeti STT v2 — whisper-MEDIUM + LoRA sur Casablanca (Algérie ×2 + Maroc aux ×1),
# merge → CTranslate2 int8, eval vs le modèle SERVI actuel (khidmeti-stt v1 = small FT 0.733),
# push HF privé seulement si meilleur. Généré par stt_push.py (HF_TOKEN injecté). Kaggle T4.
#
# Pourquoi medium+LoRA : small full-FT a plafonné à WER 0.733 (760 clips) ; DZIRI VoiceBOT
# (arXiv 2606.26003) = medium meilleur backbone pour l'algérien ; LoRA régularise sur si peu
# de données et tient sur T4 (base gelée + checkpointing). Maroc = darija sœur, val/test 100% DZ.
#
# Licence dataset: CC BY-NC-ND 4.0 — poids intérimaires (usage étudiant/pré-lancement) ;
# remplacés par le fine-tune flywheel P6 (données propres) avant toute commercialisation.
import gc, json, os, random, re, subprocess, sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "datasets==3.6.0", "soundfile", "librosa", "jiwer",
                "ctranslate2", "faster-whisper", "peft"], check=True)
# torchao 0.10.0 préinstallé sur Kaggle fait crasher le dispatcher LoRA de peft
# (exige ≥0.16) ; sans torchao, peft passe proprement au chemin standard.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

import numpy as np
import torch
import jiwer
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor, \
    get_linear_schedule_with_warmup

SEED = 20260804
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

HF_TOKEN = "{{HF_TOKEN}}"
os.environ["HF_TOKEN"] = HF_TOKEN  # huggingface_hub le lit pour le repo privé
REPO   = "Walidrbh27/khidmeti-stt"
MEDIUM = "openai/whisper-medium"
EPOCHS, LR, BATCH, ACCUM = 5, 1e-4, 4, 4   # LR LoRA standard ; eff. batch 16
GEN_BS, MAX_LABEL = 8, 448
DEV = "cuda"

# ---- normalisation arabe pour WER (identique v1 — comparabilité) ----
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
except Exception as e:  # dégradation : DZ-only si le subset bouge
    print(f"WARN_NO_MOROCCO_AUX: {e}")
train = train_dz * 2 + aux_ma  # DZ sur-échantillonné ×2, val/test 100% algériens
print(f"train_dz={len(train_dz)} aux_ma={len(aux_ma)} train={len(train)} "
      f"val={len(val)} test={len(test)}")

processor = WhisperProcessor.from_pretrained(MEDIUM, language="Arabic", task="transcribe")
tok = processor.tokenizer
START = tok.convert_tokens_to_ids("<|startoftranscript|>")

def collate(batch, with_labels=True):
    feats = processor([b["audio"]["array"] for b in batch],
                      sampling_rate=16000, return_tensors="pt").input_features
    if not with_labels:
        return feats
    enc = tok([b["transcription"] for b in batch], return_tensors="pt",
              padding=True, truncation=True, max_length=MAX_LABEL)
    # masque UNIQUEMENT le padding (attention_mask) — pad == eot chez whisper
    labs = enc.input_ids.masked_fill(enc.attention_mask.ne(1), -100)
    if (labs[:, 0] == START).all():   # le modèle ré-ajoute le start token (shift_right)
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
    return w

# ---- baseline = le modèle SERVI aujourd'hui (khidmeti-stt v1), int8, full test ----
from huggingface_hub import snapshot_download, HfApi
served_dir = snapshot_download(REPO, token=HF_TOKEN)
served_wer = fw_eval(served_dir, test, "BASELINE_SERVED_V1")

# ---- medium + LoRA ----
from peft import LoraConfig, get_peft_model, \
    get_peft_model_state_dict, set_peft_model_state_dict

base = WhisperForConditionalGeneration.from_pretrained(MEDIUM)
base.generation_config.forced_decoder_ids = None
base.generation_config.use_cache = True
cfg = base.config
cfg.use_cache = False               # teacher forcing, silence le warning checkpointing
cfg.apply_spec_augment = True
cfg.mask_time_prob = 0.05
cfg.mask_feature_prob = 0.0
base.model.encoder.conv1.register_forward_hook(
    lambda m, i, o: o.requires_grad_(True))  # base gelée → il faut un grad d'entrée
base.gradient_checkpointing_enable()

model = get_peft_model(base, LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])).to(DEV)
model.print_trainable_parameters()

opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
steps = (len(train) // (BATCH * ACCUM) + 1) * EPOCHS
sched = get_linear_schedule_with_warmup(opt, int(steps * 0.1), steps)
scaler = torch.amp.GradScaler("cuda")

best_wer, best_adapter = float("inf"), None
order = list(range(len(train)))
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
    cfg.apply_spec_augment = False
    vw, _ = wer_cer([x["transcription"] for x in val],
                    transcribe_all(model, val, f"val_ep{ep+1}"))
    cfg.apply_spec_augment = True
    print(f"epoch {ep+1} loss={tot/max(nb,1):.4f} val_WER={vw:.4f}")
    if vw < best_wer:
        best_wer = vw
        best_adapter = {k: v.detach().cpu().clone()
                        for k, v in get_peft_model_state_dict(model).items()}

cfg.apply_spec_augment = False
set_peft_model_state_dict(model, best_adapter)

# ---- merge LoRA → poids pleins, export CT2 int8 ----
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
new_wer = fw_eval("ct2_out", test, "FT_MEDIUM_INT8")

# ---- gate : doit battre le modèle SERVI, sinon rien n'est poussé ----
print(f"GATE_BEATS_SERVED={'YES' if new_wer < served_wer else 'NO'} "
      f"(new={new_wer:.4f} vs served={served_wer:.4f})")
if new_wer >= served_wer:
    print("GATE_FAILED_NO_UPLOAD — medium LoRA ne bat pas le small FT servi")
    sys.exit(0)

json.dump({"base_model": MEDIUM, "method": "LoRA r32 a64 qkvo+fc merged",
           "dataset": "UBC-NLP/Casablanca:Algeria x2 + Morocco aux (CC BY-NC-ND 4.0)",
           "license_note": "poids intérimaires non-commerciaux — remplacer via flywheel P6",
           "trained": "2026-08-04", "seed": SEED,
           "train_dz": len(train_dz), "aux_ma": len(aux_ma),
           "val_rows": len(val), "test_rows": len(test),
           "wer_norm": {"served_v1_int8": served_wer, "ft_medium_int8": new_wer,
                        "best_val": best_wer},
           "serve": "faster-whisper compute_type=int8 language=ar"},
          open("ct2_out/meta.json", "w"), indent=2)
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

