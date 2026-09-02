# ml/kaggle/vision_kernel.py — Kaggle script kernel (CPU, internet ON)
#
# P5 pipeline darija — export SigLIP2 zero-shot pour ai-vision :
#   1. charge google/siglip2-base-patch16-224 (fp32, CPU)
#   2. encode les prompts anglais "photo de panne" (16 métiers P2b + none)
#      → text_embeds.npy (L2-norm)
#   3. exporte la tour IMAGE seule en ONNX (opset 17, dynamo=False — leçon P2)
#      + quantize_dynamic int8 ; la tour texte ne sert plus au runtime
#   4. push le tout dans le repo HF privé Walidrbh27/khidmeti-vision
#
# Pas de gate d'accuracy ici (aucune photo d'éval) : les asserts couvrent la
# parité preprocessing (manuel PIL == processor HF, répliqué dans server.py)
# et la parité ONNX (cos torch↔fp32 ↔ int8). Le test sémantique = photo de
# tuyau cassé sur la machine de build (critère de fin P5).
#
# Placeholders injectés par vision_push.py : {{HF_TOKEN}} {{LABELS_B64}}

import base64, gzip, json, subprocess, sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "onnx", "onnxruntime", "transformers>=4.53"], check=True)

import numpy as np
import torch
from PIL import Image

CKPT = "google/siglip2-base-patch16-224"
REPO = "Walidrbh27/khidmeti-vision"
HF_TOKEN = "{{HF_TOKEN}}"
LABELS = json.loads(gzip.decompress(base64.b64decode("{{LABELS_B64}}")))

# ── Prompts zero-shot — l'utilisateur photographie la PANNE (leçon P1), en
# anglais (chemin le plus balisé pour SigLIP). 'none' = distracteurs.
PROMPTS = {
    "plumber": [
        "a photo of a burst water pipe leaking",
        "a photo of a leaking faucet dripping water",
        "a photo of a clogged blocked sink drain",
        "a photo of a water leak under a kitchen sink",
        "a photo of water stains on a ceiling from a leak",
        "a photo of a broken toilet or leaking water heater",
    ],
    "electrician": [
        "a photo of exposed electrical wires",
        "a photo of a burnt scorched power outlet",
        "a photo of an electrical panel with circuit breakers",
        "a photo of a broken light switch or ceiling light fixture",
        "a photo of tangled electrical cables on a wall",
    ],
    "ac_repair": [
        "a photo of a wall mounted split air conditioner unit",
        "a photo of an air conditioner leaking water",
        "a photo of an air conditioner outdoor compressor unit",
        "a photo of a dirty air conditioner filter",
    ],
    "mason": [
        "a photo of a cracked concrete wall",
        "a photo of broken wall tiles or floor tiles",
        "a photo of a damaged brick wall with holes",
        "a photo of crumbling cement plaster on a wall",
        "a photo of a construction site with bricks and cement",
    ],
    "painter": [
        "a photo of peeling paint on a wall",
        "a photo of a stained discolored wall needing repainting",
        "a photo of an unpainted grey plastered wall",
        "a photo of paint cans and rollers in a room",
    ],
    "carpenter": [
        "a photo of a broken wooden door",
        "a photo of broken wooden furniture",
        "a photo of a damaged wooden cabinet or wardrobe",
        "a photo of a broken window frame",
        "a photo of wooden planks and carpentry tools",
    ],
    "cleaner": [
        "a photo of a very dirty messy room",
        "a photo of a dirty kitchen with grease and grime",
        "a photo of a stained dirty carpet or sofa",
        "a photo of a dirty bathroom needing cleaning",
        "a photo of a dusty cluttered apartment",
    ],
    "appliance_repair": [
        "a photo of a washing machine",
        "a photo of a refrigerator or fridge",
        "a photo of a broken oven or kitchen stove",
        "a photo of a dishwasher or microwave oven",
    ],
    "mover": [
        "a photo of moving boxes stacked in a room",
        "a photo of furniture wrapped in plastic for moving house",
        "a photo of a moving truck being loaded with furniture",
        "a photo of packed cardboard boxes for relocation",
    ],
    "mechanic": [
        "a photo of a car engine bay",
        "a photo of a broken down car with the hood open",
        "a photo of a flat car tire",
        "a photo of a car dashboard with warning lights",
        "a photo of a car being repaired in a garage",
    ],
    "plasterer": [
        "a photo of a suspended ceiling with drywall panels",
        "a photo of drywall plasterboard sheets being installed",
        "a photo of a damaged sagging plasterboard ceiling",
        "a photo of metal stud framing for a drywall partition",
        "a photo of a decorative gypsum ceiling with spotlights",
    ],
    "welder": [
        "a photo of a wrought iron gate or metal door",
        "a photo of a broken metal fence or railing",
        "a photo of welding metal with sparks",
        "a photo of a metal window security grille",
        "a photo of a rusty broken metal structure",
    ],
    "barber": [
        "a photo of a men's haircut or hairstyle",
        "a photo of a barbershop interior with a barber chair",
        "a photo of hair clippers and barber tools",
        "a photo of a beard trim or fade haircut",
    ],
    "tailor": [
        "a photo of torn or ripped clothing",
        "a photo of a sewing machine",
        "a photo of a broken zipper on a garment",
        "a photo of fabric rolls and sewing tools",
        "a photo of a traditional dress or caftan",
    ],
    "caterer": [
        "a photo of a couscous platter or traditional North African dish",
        "a photo of catering trays with food for a party",
        "a photo of traditional pastries and sweets on a table",
        "a photo of a buffet table at a wedding",
    ],
    # 'none' : plus de distracteur nourriture — une photo de plat est désormais
    # un signal caterer (P2b), pas du bruit. v3.1 : le jardin devient un
    # distracteur (gardener retiré du catalogue).
    "none": [
        "a photo of a person's face, a selfie",
        "a screenshot of a phone app with text",
        "a photo of a document or paper with text",
        "a photo of a landscape or street scene",
        "a photo of a cat or a dog",
        "a photo of a garden lawn with plants and trees",
    ],
}
assert set(PROMPTS) == set(LABELS["professions"]), "prompts ≠ labels.json gelé"

from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

model = AutoModel.from_pretrained(CKPT).eval()
tok = AutoTokenizer.from_pretrained(CKPT)
try:  # processor lent (PIL) — server.py réplique PIL, pas torchvision
    iproc = AutoImageProcessor.from_pretrained(CKPT, use_fast=False)
except Exception:
    iproc = AutoImageProcessor.from_pretrained(CKPT)

# ── Text embeddings (une fois, offline — la tour texte meurt ici) ────────────
classes = list(PROMPTS)
texts, prompt_class = [], []
for i, c in enumerate(classes):
    for p in PROMPTS[c]:
        texts.append(p)
        prompt_class.append(i)

def feat(out):
    # transformers récents : get_*_features rend un ModelOutput, pas un tenseur
    return out if torch.is_tensor(out) else out.pooler_output

enc = tok(texts, padding="max_length", max_length=64, truncation=True,
          return_tensors="pt")
with torch.no_grad():
    temb = feat(model.get_text_features(**enc))
temb = torch.nn.functional.normalize(temb, dim=-1).numpy().astype(np.float32)
scale = float(model.logit_scale.exp().item())
bias = float(model.logit_bias.item())
print(f"text_embeds {temb.shape}  logit_scale={scale:.3f} logit_bias={bias:.3f}")

# ── Parité preprocessing : manuel (répliqué dans server.py) == processor HF ──
H = int(iproc.size["height"]); W = int(iproc.size["width"])
MEAN = np.array(iproc.image_mean, dtype=np.float32)
STD = np.array(iproc.image_std, dtype=np.float32)
RESAMPLE = {0: "nearest", 2: "bilinear", 3: "bicubic"}[int(iproc.resample)]

def manual_preprocess(img):
    img = img.convert("RGB").resize((W, H), getattr(Image, RESAMPLE.upper()))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None]

rng = np.random.default_rng(20260803)
test_img = Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))
ref = iproc(images=test_img, return_tensors="np").pixel_values
diff = float(np.abs(manual_preprocess(test_img) - ref).max())
print(f"preprocess parity max|Δ| = {diff:.5f}  ({W}x{H} {RESAMPLE})")
assert diff < 2e-2, "preprocessing manuel ≠ processor HF — server.py serait faux"

# ── Export ONNX tour image (embedding L2-normalisé en sortie) ────────────────
class VisionEmbed(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        e = self.m.get_image_features(pixel_values=pixel_values)
        if not torch.is_tensor(e):
            e = e.pooler_output
        return e / e.norm(dim=-1, keepdim=True)

wrapper = VisionEmbed(model).eval()
dummy = torch.from_numpy(manual_preprocess(test_img))
torch.onnx.export(
    wrapper, (dummy,), "model.fp32.onnx",
    input_names=["pixel_values"], output_names=["image_embeds"],
    dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}},
    opset_version=17, dynamo=False,  # leçon P2 : dynamo exige onnxscript
)

from onnxruntime.quantization import QuantType, quantize_dynamic
quantize_dynamic("model.fp32.onnx", "model.int8.onnx",
                 weight_type=QuantType.QInt8, per_channel=True)

# ── Parité ONNX : torch ↔ fp32 ↔ int8 sur 4 images aléatoires ────────────────
import onnxruntime as ort
batch = np.concatenate([
    manual_preprocess(Image.fromarray(
        rng.integers(0, 255, (240 + 40 * i, 320, 3), dtype=np.uint8)))
    for i in range(4)])
with torch.no_grad():
    ref_t = wrapper(torch.from_numpy(batch)).numpy()

def cos(a, b):
    return float(np.mean(np.sum(a * b, axis=-1) /
                         (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))))

out32 = ort.InferenceSession("model.fp32.onnx").run(None, {"pixel_values": batch})[0]
out8 = ort.InferenceSession("model.int8.onnx").run(None, {"pixel_values": batch})[0]
c32, c8 = cos(ref_t, out32), cos(out32, out8)
# Le kernel DÉCIDE le fichier servi : int8 seulement s'il préserve les
# embeddings (v2 per-tensor : cos 0.934 → inutilisable en zero-shot, les
# marges inter-classes sont du même ordre). fp32 tient dans 1g de RAM.
serve_file = "model.int8.onnx" if c8 > 0.99 else "model.fp32.onnx"
print(f"cos torch↔fp32 = {c32:.5f}   cos fp32↔int8 = {c8:.5f}   → {serve_file}")
assert c32 > 0.999, "export ONNX fp32 divergent"

# smoke scoring (formes seulement — la sémantique se juge sur machine de build)
out_s = out8 if serve_file == "model.int8.onnx" else out32
logits = scale * (out_s[:1] @ temb.T) + bias
per_cls = [max(l for l, ci in zip(logits[0], prompt_class) if ci == i)
           for i in range(len(classes))]
top = sorted(zip(classes, per_cls), key=lambda t: -t[1])[:3]
print("smoke top-3 (image aléatoire):",
      [(c, round(1 / (1 + np.exp(-l)), 4)) for c, l in top])

# ── Push HF ──────────────────────────────────────────────────────────────────
np.save("text_embeds.npy", temb)
with open("vision_labels.json", "w") as f:
    json.dump({"classes": classes, "prompts": texts,
               "prompt_class": prompt_class}, f, ensure_ascii=False, indent=1)
with open("meta.json", "w") as f:
    json.dump({
        "checkpoint": CKPT, "image_size": [H, W],
        "image_mean": MEAN.tolist(), "image_std": STD.tolist(),
        "resample": RESAMPLE, "text_max_len": 64, "dim": int(temb.shape[1]),
        "logit_scale": scale, "logit_bias": bias, "opset": 17,
        "cos_torch_fp32": c32, "cos_fp32_int8": c8, "serve_file": serve_file,
    }, f, indent=1)

from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)
api.create_repo(REPO, private=True, exist_ok=True)
for fname in ["model.int8.onnx", "model.fp32.onnx", "text_embeds.npy",
              "vision_labels.json", "meta.json"]:
    api.upload_file(path_or_fileobj=fname, path_in_repo=fname, repo_id=REPO)
    print("uploaded", fname)
print("ALL DONE —", REPO)

# Auto-restart Space to pick up new weights
try:
    from huggingface_hub import HfApi as _HfApi2
    _HfApi2(token=HF_TOKEN).restart_space(repo_id="Walidrbh27/khidmeti-ai", repo_type="space")
    print("SPACE_RESTARTED=Walidrbh27/khidmeti-ai")
except Exception as _e:
    print(f"SPACE_RESTART_SKIP={_e}")

