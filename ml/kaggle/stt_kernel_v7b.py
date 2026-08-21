#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
# ml/kaggle/stt_kernel_v7b.py — reprise du run v7 (whisper-large-v3 + LoRA, DZ)
#
# v7 a entraîné 5 epochs proprement (best val WER 0,5738 @ epoch 3, 0 step
# non-fini) puis est mort sur la PREMIÈRE eval de test :
#   ValueError: expected shape (1, 128, 3000), got (1, 80, 3000)
# large-v3 utilise 128 bandes mel (80 pour medium/small) et faster-whisper lit
# ce nombre UNIQUEMENT dans preprocessor_config.json ; ce fichier n'a pas
# atteint ct2_out, donc extraction à 80 face à un modèle à 128.
#
# Récupération sans réentraîner : ÉCHEC. Kaggle n'expose que la sortie de la
# dernière version TERMINÉE d'un kernel ; la version v7 étant en erreur, la
# source montée renvoyait la sortie du run v3 du 4 août (medium, 80 mels —
# l'erreur s'était inversée en « expected 80, got 128 », et ct2_out contenait
# meta.json que v7 n'écrit qu'APRÈS les evals). Poids v7 perdus.
#
# Deux leçons câblées ici :
#   1. preprocessor_config.json copié EXPLICITEMENT dans ct2_out + assert 128
#      + smoke eval 3 clips juste après l'export → échoue en 1 min, pas en 6 h.
#   2. UPLOAD DU CANDIDAT SUR HF DÈS L'EXPORT, avant les evals, dans un dépôt
#      séparé (pas de sous-dossier dans REPO : ai-stt fait snapshot_download de
#      la racine et tirerait 1,6 Go en trop en prod). Un crash en phase d'eval
#      ne doit plus jamais coûter l'entraînement.
#
# Déjà mesuré (run v7, MÊME test set 831 clips, MÊME moteur int8) — non
# re-mesuré ici, d'où ~6 h au lieu de 9 :
#   BASELINE_SERVED_V3  WER 0,6300  CER 0,2373
#   ZEROSHOT_LARGE_V3   WER 0,8055  CER 0,3708   ← le socle nu est PIRE : tout
#                                                  le gain vient du LoRA
# ══════════════════════════════════════════════════════════════════════════════
import gc, json, os, random, re, shutil, subprocess, sys, time

T0 = time.time()
DEADLINE_S = 7.6 * 3600          # ponytail: budget mur ; au-delà on exporte quand même

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
from huggingface_hub import snapshot_download, HfApi
SEED = 20260814                                  # identique v7 : même split, courbe comparable
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

HF_TOKEN = "{{HF_TOKEN}}"
os.environ["HF_TOKEN"] = HF_TOKEN
REPO       = "Walidrbh27/khidmeti-stt"
CAND_REPO  = "Walidrbh27/khidmeti-stt-cand"      # filet : poids sauvés avant les evals
BASE_MODEL = "openai/whisper-large-v3"
EPOCHS, LR, BATCH, ACCUM = 4, 3e-5, 1, 16        # v7 : best epoch 3, 4/5 ne gagnaient plus
LORA_R, LORA_A = 32, 64
GEN_BS, MAX_LABEL = 2, 448
DEV = "cuda"
MELS = 128                                       # large-v3 ; 80 pour medium/small

SERVED_WER, SERVED_CER = 0.6300, 0.2373          # run v7, même test set, même moteur
ZS_WER,     ZS_CER     = 0.8055, 0.3708

# Amorce métier IDENTIQUE à docker/ai-stt/server.py (_DEFAULT_PROMPT) — pour
# mesurer le WER tel que la prod le produit réellement, pas seulement à nu.
PROD_PROMPT = (
    "خاصني بلومبي كهربائي سباك نجار حداد سودور بناي ماصون صباغ حلاق كوافور "
    "خياط طباخة جباص بلاكو ميكانيسيان كليماتيزور فريجيدار ماشينة تنظيف "
    "ديمناجمون تصليح نحوس على واحد يجيني للدار"
)

# ---- normalisation arabe pour WER (identique v3/v6/v7 — comparabilité stricte) ----
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
assert len(test) == 831, f"test set différent du run v7 : {len(test)}"   # comparabilité

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

# LE bug qui a tué v7 : faster-whisper prend le nombre de bandes mel dans
# preprocessor_config.json et NULLE PART ailleurs. On ne fait plus confiance à
# ce que save_pretrained a bien voulu écrire — on copie depuis le socle et on
# vérifie.
def to_ct2(hf_dir, out_dir):
    from ctranslate2.converters import TransformersConverter
    TransformersConverter(hf_dir, copy_files=["tokenizer.json"]).convert(
        out_dir, quantization="int8", force=True)
    pp = snapshot_download(BASE_MODEL, token=HF_TOKEN,
                          allow_patterns=["preprocessor_config.json"])
    shutil.copy(f"{pp}/preprocessor_config.json", f"{out_dir}/preprocessor_config.json")
    got = json.load(open(f"{out_dir}/preprocessor_config.json"))["feature_size"]
    assert got == MELS, f"preprocessor_config.json annonce {got} mels, attendu {MELS}"
    print(f"{out_dir} ok : {sorted(os.listdir(out_dir))}", flush=True)
    return out_dir

# ---- large-v3 + LoRA r32 ----
from peft import LoraConfig, get_peft_model, \
    get_peft_model_state_dict, set_peft_model_state_dict

print(f"=== large-v3 + LoRA r{LORA_R} — {left()/3600:.1f} h restantes ===", flush=True)
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
        if left() < 4200:                 # 70 min : export + upload candidat + 3 evals
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

# ---- merge LoRA → fp16 → export CT2 int8 ----
merged = model.merge_and_unload().cpu().half()   # fp16 : 3,1 Go au lieu de 6,2 sur disque
cfg.use_cache = True
merged.save_pretrained("ft_hf"); processor.save_pretrained("ft_hf")
del model, merged, base; gc.collect(); torch.cuda.empty_cache()
to_ct2("ft_hf", "ct2_out")
shutil.rmtree("ft_hf", ignore_errors=True)
# ---- smoke : 3 clips. Si l'export est cassé on le sait en 1 min, pas en 6 h ----
sw, _ = fw_eval("ct2_out", test[:3], "SMOKE")
assert sw < 1.5, f"export suspect, WER smoke {sw:.2f}"

# ---- FILET : poids sur HF avant toute eval longue. Un crash d'eval ne doit
#      plus coûter l'entraînement (v7 : 6 h perdues exactement comme ça). ----
api = HfApi(token=HF_TOKEN)
api.create_repo(CAND_REPO, private=True, exist_ok=True)
api.upload_folder(folder_path="ct2_out", repo_id=CAND_REPO)
print(f"CANDIDATE_SAVED={CAND_REPO} (best_val_WER={best_wer:.4f})", flush=True)

# ---- evals de test : à nu ET avec l'amorce métier que la prod passe vraiment ----
print(f"=== evals finales — {left()/3600:.1f} h restantes ===", flush=True)
new_wer,  new_cer  = fw_eval("ct2_out", test, "FT_LARGE_V3_V7")
newp_wer, newp_cer = fw_eval("ct2_out", test, "FT_LARGE_V3_V7_PROMPTED", prompt=PROD_PROMPT)
servedp_wer, servedp_cer = fw_eval(snapshot_download(REPO, token=HF_TOKEN),
                                   test, "BASELINE_SERVED_V3_PROMPTED", prompt=PROD_PROMPT)

print("\n┌─ récap (même test set 831 clips, même moteur faster-whisper int8) ─")
print(f"│ servi v3 medium+LoRA   à nu {SERVED_WER:.4f}   amorce {servedp_wer:.4f}")
print(f"│ large-v3 zero-shot     à nu {ZS_WER:.4f}")
print(f"│ large-v3 +LoRA (v7)    à nu {new_wer:.4f}   amorce {newp_wer:.4f}")
print(f"│ apport du socle seul : {SERVED_WER - ZS_WER:+.4f}")
print(f"│ apport du LoRA       : {ZS_WER - new_wer:+.4f}")
print(f"│ apport de l'amorce   : v3 {SERVED_WER - servedp_wer:+.4f}  v7 {new_wer - newp_wer:+.4f}")
print("└─", flush=True)

best_new, best_old = min(new_wer, newp_wer), min(SERVED_WER, servedp_wer)
gate_pass, use_p = best_new < best_old, newp_wer < new_wer
print(f"GATE_BEATS_SERVED={'YES' if gate_pass else 'NO'} "
      f"(v7 {best_new:.4f} vs servi {best_old:.4f})")
print(f"INFO_WER_55={'YES' if best_new < 0.55 else 'NO'}")
print(f"PROMPT_HELPS_V7={'YES' if use_p else 'NO'} → STT_PROMPT={'on' if use_p else 'off'}")
print(f"GATE_PASS={'YES' if gate_pass else 'NO'}", flush=True)

if not gate_pass:
    print(f"GATE_FAILED_NO_UPLOAD — on ne remplace rien ; poids gardés sur {CAND_REPO}")
    sys.exit(0)

# ---- où le modèle casse (clips courts = notre cas d'usage réel) ----
print("\nWER par tranche de durée :", flush=True)
strat = {}
for low, high in [(0.5, 5), (5, 10), (10, 20), (20, 30)]:
    sub = [ex for ex in test if low < ex["duration"] <= high]
    if len(sub) >= 10 and left() > 600:
        w, c = fw_eval("ct2_out", sub, f"BIN_{low}_{high}",
                       prompt=PROD_PROMPT if use_p else None)
        strat[f"{low}-{high}s"] = {"n": len(sub), "wer": w, "cer": c}
# ---- package + upload ----
json.dump({
    "version": "v7", "trained": "2026-08-15", "seed": SEED,
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
    "wer_norm": {"served_v3": SERVED_WER, "served_v3_prompted": servedp_wer,
                 "zeroshot_large_v3": ZS_WER,
                 "ft_v7": new_wer, "ft_v7_prompted": newp_wer, "best_val": best_wer},
    "cer_norm": {"served_v3": SERVED_CER, "served_v3_prompted": servedp_cer,
                 "zeroshot_large_v3": ZS_CER,
                 "ft_v7": new_cer, "ft_v7_prompted": newp_cer, "best_val": best_cer},
    "wer_by_duration": strat,
    "gate": "min(WER à nu, WER avec amorce) < même minimum du modèle servi",
    "serve": "faster-whisper compute_type=int8 language=ar",
    "serve_prompt": "on" if use_p else "off",
    "warning_mem_limit": "large-v3 int8 ≈ 1,6 Go de poids : docker-compose ai-stt "
                         "mem_limit doit passer de 2g à 4g",
    "note_128_mels": "large-v3 utilise 128 bandes mel (80 en dessous) : "
                     "preprocessor_config.json est OBLIGATOIRE dans le dossier CT2, "
                     "sinon faster-whisper retombe sur 80 et lève une ValueError de forme",
}, open("ct2_out/meta.json", "w"), indent=2, ensure_ascii=False)

api.upload_folder(folder_path="ct2_out", repo_id=REPO)
print("HF_UPLOAD_DONE=" + REPO)
print(f"V7 COMPLETE: WER {best_new:.4f} (servi {best_old:.4f}) "
      f"en {(time.time()-T0)/3600:.1f} h")
