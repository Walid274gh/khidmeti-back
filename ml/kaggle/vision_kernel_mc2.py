# ══════════════════════════════════════════════════════════════════════════════
# ml/kaggle/vision_kernel_mc2.py — Kaggle script kernel (CPU, internet ON)
#
# P5c — MobileCLIP2 face au SigLIP2 servi, MESURÉ SUR LES MÊMES IMAGES.
#
# Pourquoi : ai-vision sert google/siglip2-base-patch16-224 en ONNX fp32 (l'int8
# tuait les embeddings : cos 0,9408) — 86 M de paramètres de tour image, ~150 ms
# par photo sur le CPU 4 threads de la machine d'auto-hébergement. MobileCLIP2
# (Apple, 2025) atteint des scores CLIP comparables avec une tour FastViT
# réparamétrable 5 à 10× plus légère. Si la précision tient, c'est tout gagné.
#
# Ce kernel fait le travail COMPLET pour chaque tour, sur UNE SEULE collecte
# d'images (sinon la comparaison ne vaut rien) :
#   1. export ONNX fp32 + int8 de la tour image (L2-norm cuite dans le graphe),
#      parité torch↔fp32↔int8 mesurée, serve_file décidé par la mesure
#   2. encodage des 76 prompts (15 métiers P2b + none) par la tour texte
#   3. collecte ddgs UNE FOIS → octets JPEG en RAM → décodés par chaque tour
#      avec SA propre transformation native (parité assertée contre open_clip)
#   4. zero-shot, prototypes k-means, raffinement BCE, Platt — protocole
#      IDENTIQUE à vision_probe_kernel.py (P5b v2) pour les trois tours
#   5. latence CPU 4 threads par tour = le vrai argument de MobileCLIP2
#
# Le SigLIP2 servi n'est JAMAIS touché : les artefacts MobileCLIP2 partent dans
# un repo NEUF. Bascule = deux variables d'env, retour arrière immédiat.
#
# Leçon v6 câblée ici : aucun modèle n'est adopté sur sa fiche HF. On mesure sur
# nos images, avec notre moteur de service, contre le modèle en production.
#
# Placeholder injecté par le push : {{HF_TOKEN}}
# ══════════════════════════════════════════════════════════════════════════════
import io, json, os, random, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                # planchers réels : MobileCLIP2 = fastvit_mci* (timm 1.0.9+) chargé
                # depuis open_clip_config.json du hub (open_clip 2.30+, custom_text
                # + hf_tokenizer_name). Sans plancher, une image Kaggle ancienne
                # échouerait au chargement 40 min après le début du kernel.
                "open_clip_torch>=2.30", "timm>=1.0.15", "ddgs", "onnx",
                "onnxruntime", "pillow", "requests", "transformers>=4.53"],
               check=True)

import numpy as np
import requests
import torch
from PIL import Image

SEED = 20260817
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(4)                    # = STT_THREADS/serveur auto-hébergé

HF_TOKEN     = "{{HF_TOKEN}}"
SIGLIP_REPO  = "Walidrbh27/khidmeti-vision"       # servi — lu, jamais écrit
SIGLIP_CKPT  = "google/siglip2-base-patch16-224"  # tour texte (perdue du repo)
MC2_REPO     = "Walidrbh27/khidmeti-vision-mc2"   # neuf
MC2_HUBS     = {"mc2_s2": "hf-hub:timm/MobileCLIP2-S2-OpenCLIP",
                "mc2_s0": "hf-hub:timm/MobileCLIP2-S0-OpenCLIP"}
PER_CLASS_CAP = 400        # images valides max/classe (identique P5b v2)
MIN_TRAIN     = 40         # en dessous : la classe garde ses prompts texte seuls
GATE_MAX_LOSS = 0.01       # bascule si MC2 perd ≤1 pt macro contre le servi
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "Chrome/126 Safari/537.36"}

# ── Prompts zero-shot — COPIE EXACTE de vision_kernel.py (kernel autonome :
# Kaggle ne pousse qu'un fichier, pas d'import possible). L'utilisateur
# photographie la PANNE (leçon P1) ; anglais = chemin le plus balisé.
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
    "none": [
        "a photo of a person's face, a selfie",
        "a screenshot of a phone app with text",
        "a photo of a document or paper with text",
        "a photo of a landscape or street scene",
        "a photo of a cat or a dog",
        "a photo of a garden lawn with plants and trees",
    ],
}
CLASSES = list(PROMPTS)
TEXTS, PROMPT_CLASS = [], []
for _i, _c in enumerate(CLASSES):
    for _p in PROMPTS[_c]:
        TEXTS.append(_p); PROMPT_CLASS.append(_i)
# garde-fou structurel, pas un compte figé (P5 en avait 57, P2b en a 76) : ce qui
# doit rester vrai, c'est un prompt → une classe, et aucune classe muette.
assert len(TEXTS) == len(PROMPT_CLASS) and set(PROMPT_CLASS) == set(range(len(CLASSES)))
print(f"prompts: {len(TEXTS)} pour {len(CLASSES)} classes", flush=True)

# ── Requêtes de collecte — COPIE EXACTE de vision_probe_kernel.py (P5b v2) :
# mêmes requêtes = collecte comparable au probe SigLIP2 d'août.
QUERIES = {
    "plumber": ["fuite d'eau tuyau lavabo", "robinet qui fuit", "évier bouché débouchage",
                "water leak under sink", "burst pipe leaking", "chauffe-eau qui fuit",
                "toilette bouchée déborde", "تسريب ماء حنفية", "أنابيب مياه تسريب",
                "fuite d'eau plafond dégât des eaux", "siphon évier démonté plomberie"],
    "electrician": ["prise électrique brûlée", "fils électriques dénudés mur",
                    "tableau électrique disjoncteurs", "burnt electrical outlet",
                    "exposed electrical wires wall", "installation lustre plafond câbles",
                    "كهرباء منزل أسلاك", "لوحة كهرباء منزلية",
                    "court-circuit prise noircie", "compteur électrique disjoncté"],
    "ac_repair": ["climatiseur split mural", "climatiseur qui fuit eau",
                  "unité extérieure climatiseur", "air conditioner leaking water",
                  "dirty air conditioner filter", "climatiseur en panne réparation",
                  "مكيف هواء تسريب ماء", "تصليح مكيف سبليت",
                  "climatiseur givré glace", "gaz climatiseur recharge manomètre"],
    "mason": ["mur fissuré fissure", "carrelage cassé sol", "mur brique trou chantier",
              "cracked concrete wall", "broken floor tiles", "travaux maçonnerie parpaing",
              "dalle béton fissurée", "جدار متشقق شقوق", "بناء جدار طوب",
              "escalier béton cassé", "mur effondré gravats"],
    "painter": ["peinture écaillée mur", "mur taché humidité", "peeling paint wall",
                "wall needs repainting", "mur à repeindre travaux peinture",
                "moisissure mur peinture cloquée", "طلاء جدران متقشر", "دهان حائط رطوبة",
                "plafond taché infiltration peinture", "façade peinture abîmée"],
    "carpenter": ["porte en bois cassée", "meuble en bois cassé", "armoire bois abîmée",
                  "broken wooden door", "broken wooden furniture", "fenêtre bois abîmée",
                  "montage meuble cuisine bois menuiserie", "باب خشب مكسور", "نجارة خشب ورشة",
                  "placard bois monté menuisier", "parquet bois abîmé"],
    "cleaner": ["maison très sale désordre", "cuisine sale graisse", "salle de bain sale calcaire",
                "very dirty messy room", "dirty kitchen grease grime", "tapis taché sale",
                "appartement poussiéreux ménage", "تنظيف منزل متسخ", "مطبخ متسخ دهون",
                "vitres sales nettoyage", "matelas taché nettoyage"],
    "appliance_repair": ["machine à laver en panne réparation", "réfrigérateur en panne",
                         "four électrique cassé", "washing machine repair inside",
                         "broken refrigerator repair", "lave-vaisselle en panne",
                         "micro-ondes cassé réparation", "غسالة معطلة تصليح", "ثلاجة معطلة",
                         "réparation électroménager technicien",
                         "machine à laver démontée réparation"],
    "mover": ["cartons déménagement empilés", "camion déménagement meubles",
              "meubles emballés plastique déménagement", "moving boxes stacked room",
              "moving truck loading furniture", "déménagement appartement cartons",
              "نقل أثاث شاحنة", "عفش نقل كراتين",
              "monte-meuble déménagement", "furniture wrapped blankets moving"],
    "mechanic": ["moteur voiture en panne", "voiture capot ouvert panne", "pneu crevé voiture",
                 "car engine bay problem", "flat tire car", "voiture en panne bord route",
                 "tableau de bord voyant moteur allumé", "سيارة معطلة محرك",
                 "ميكانيكي سيارات ورشة",
                 "batterie voiture à plat câbles", "fuite huile moteur voiture"],
    "plasterer": ["faux plafond placo spots", "plafond placoplatre abîmé dégât",
                  "cloison BA13 montage rails", "damaged drywall ceiling",
                  "gypsum ceiling design decoration", "plafond suspendu placo chantier",
                  "جبس بورد سقف", "ديكور جبس أسقف",
                  "plâtre mur abîmé trous", "ba13 cloison chantier rails"],
    "welder": ["portail fer forgé", "grille métallique fenêtre protection",
               "soudure métal étincelles", "wrought iron gate", "broken metal fence railing",
               "porte métallique rouillée", "rampe escalier métallique",
               "لحام حديد شرارة", "باب حديد ملحوم",
               "portail métallique cassé soudure", "structure métallique soudée"],
    "barber": ["coupe cheveux homme dégradé", "salon coiffure homme intérieur",
               "tondeuse cheveux barbier", "barbershop haircut fade", "beard trim barber",
               "coiffeur homme ciseaux", "حلاقة رجال صالون", "حلاق شعر ماكينة",
               "coiffure homme barbe dégradé", "barber shop chair vintage"],
    "tailor": ["vêtement déchiré trou", "machine à coudre couture", "fermeture éclair cassée",
               "torn ripped clothes", "sewing machine tailor work", "retouche vêtement couture",
               "caftan robe traditionnelle couture", "خياطة ملابس ماكينة", "قندورة قفطان خياطة",
               "ourlet pantalon retouche", "tissu ciseaux patron couture"],
    "caterer": ["couscous plat traditionnel", "traiteur buffet mariage",
                "gâteaux traditionnels algériens",
                "catering trays food party", "wedding buffet table food", "plats traiteur fête",
                "طعام تقليدي جزائري", "حلويات عيد تقليدية",
                "plats mariage traiteur oriental", "couscous garni viande légumes"],
    "none": ["selfie personne visage", "screenshot application téléphone",
             "document papier texte scan",
             "rue ville paysage urbain", "chat chien animal compagnie", "jardin pelouse arbres",
             "salon propre moderne intérieur", "montagne plage paysage nature",
             "groupe personnes réunion photo", "صورة شخصية سيلفي", "شاشة هاتف تطبيق",
             "facture papier administratif"],
}
assert set(QUERIES) == set(CLASSES), "requêtes ≠ classes"

# ══ 1. préprocessing : une seule implémentation, celle que server.py réplique ══
# SigLIP2 écrase l'image en carré (resize_mode "squash", son processor natif) ;
# MobileCLIP2 redimensionne le petit côté puis recadre au centre ("shortest").
# meta.json porte désormais resize_mode ; absent = "squash" → SigLIP2 inchangé.
_PIL = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC}

def make_preprocess(H, W, mean, std, resample, resize_mode):
    mean = np.asarray(mean, np.float32); std = np.asarray(std, np.float32)
    rs = _PIL[resample]

    def pp(img):
        img = img.convert("RGB")
        if resize_mode == "shortest":
            short = min(img.size)
            w = max(W, int(img.width * W / short))     # troncature = torchvision
            h = max(H, int(img.height * H / short))
            img = img.resize((w, h), rs)
            l, t = (w - W) // 2, (h - H) // 2
            img = img.crop((l, t, l + W, t + H))
        else:
            img = img.resize((W, H), rs)
        x = np.asarray(img, dtype=np.float32) / 255.0
        return ((x - mean) / std).transpose(2, 0, 1)
    return pp

import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
_OPT = ort.SessionOptions(); _OPT.intra_op_num_threads = 4   # = machine auto-hébergée

def cos(a, b):
    return float(np.mean(np.sum(a * b, -1) /
                         (np.linalg.norm(a, -1) * np.linalg.norm(b, -1))))

_rng = np.random.default_rng(SEED)
PROBES = [Image.fromarray(_rng.integers(0, 255, (240 + 37 * i, 320 + 11 * i, 3),
                                        dtype=np.uint8)) for i in range(4)]

class Emb(torch.nn.Module):
    """Sortie L2-normalisée cuite dans le graphe (le serveur ne renormalise pas)."""
    def __init__(self, m, fn):
        super().__init__(); self.m, self.fn = m, fn

    def forward(self, pixel_values):
        e = getattr(self.m, self.fn)(pixel_values)
        if not torch.is_tensor(e):
            e = e.pooler_output
        return e / e.norm(dim=-1, keepdim=True)

def export_onnx(wrapper, pp, out):
    os.makedirs(out, exist_ok=True)
    batch = np.stack([pp(im) for im in PROBES])
    with torch.no_grad():
        ref = wrapper(torch.from_numpy(batch)).numpy()
    torch.onnx.export(
        wrapper, (torch.from_numpy(batch[:1]),), f"{out}/model.fp32.onnx",
        input_names=["pixel_values"], output_names=["image_embeds"],
        dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}},
        opset_version=17, dynamo=False,          # leçon P2 : dynamo exige onnxscript
    )
    quantize_dynamic(f"{out}/model.fp32.onnx", f"{out}/model.int8.onnx",
                     weight_type=QuantType.QInt8, per_channel=True)
    run = lambda f: ort.InferenceSession(f, _OPT, providers=["CPUExecutionProvider"]) \
                       .run(None, {"pixel_values": batch})[0]
    o32, o8 = run(f"{out}/model.fp32.onnx"), run(f"{out}/model.int8.onnx")
    c32, c8 = cos(ref, o32), cos(o32, o8)
    assert c32 > 0.999, f"export fp32 divergent (cos {c32:.5f})"
    # int8 servi seulement s'il préserve les embeddings : sur SigLIP2 il ne les
    # préservait pas (cos 0,9408 → fp32 servi). Décision par la mesure, pas par foi.
    serve = "model.int8.onnx" if c8 > 0.99 else "model.fp32.onnx"
    print(f"  cos torch↔fp32={c32:.5f} cos fp32↔int8={c8:.5f} → {serve} "
          f"({os.path.getsize(f'{out}/{serve}')/1e6:.0f} Mo)", flush=True)
    return c32, c8, serve

def load_mc2(name, hub):
    import open_clip
    print(f"\n=== {name} : {hub} ===", flush=True)
    model, _, pp_ref = open_clip.create_model_and_transforms(hub)
    model.eval()
    tokz = open_clip.get_tokenizer(hub)
    try:
        cfg = open_clip.get_model_preprocess_cfg(model)
    except Exception:
        cfg = {}
    size = cfg.get("size", (256, 256))
    size = (size, size) if isinstance(size, int) else tuple(size)[:2]
    mean = tuple(cfg.get("mean", (0.0, 0.0, 0.0)))
    std = tuple(cfg.get("std", (1.0, 1.0, 1.0)))
    interp = str(cfg.get("interpolation", "bilinear"))
    rmode = str(cfg.get("resize_mode", "shortest"))
    pp = make_preprocess(size[0], size[1], mean, std, interp, rmode)
    print(f"  preprocess {size} {interp} {rmode} mean={mean} std={std}", flush=True)

    with torch.no_grad():
        t = model.encode_text(tokz(TEXTS))
    temb = torch.nn.functional.normalize(t, dim=-1).numpy().astype(np.float32)
    lb = getattr(model, "logit_bias", None)
    scale = float(model.logit_scale.exp().item())
    bias = float(lb.item()) if lb is not None else 0.0

    # parité : mon préprocessing (celui de server.py) == transform open_clip
    d = float(np.abs(pp(PROBES[0]) - pp_ref(PROBES[0]).numpy()).max())
    print(f"  parité preprocess max|Δ|={d:.5f}", flush=True)
    assert d < 2e-2, "préprocessing manuel ≠ open_clip — server.py serait faux"

    # FastViT est réparamétrable : les branches d'entraînement fusionnent en
    # convolutions simples. C'est LÀ que MobileCLIP gagne sa vitesse.
    wrap = Emb(model, "encode_image").eval()
    with torch.no_grad():
        before = wrap(torch.from_numpy(np.stack([pp(im) for im in PROBES]))).numpy()
    orig_visual = model.visual          # reparameterize_model deepcopy → intact
    try:
        from timm.utils import reparameterize_model
        model.visual = reparameterize_model(model.visual)
        wrap = Emb(model, "encode_image").eval()
        with torch.no_grad():
            after = wrap(torch.from_numpy(np.stack([pp(im) for im in PROBES]))).numpy()
        c = cos(before, after)
        assert c > 0.995, f"cos {c:.5f}"
        print(f"  réparamétrisé (cos {c:.5f})", flush=True)
    except Exception as e:
        # la fusion est une optimisation de vitesse, pas la mesure : on repart du
        # modèle non fusionné plutôt que de perdre le run (la latence mesurée
        # sera pessimiste, et le WARN le dit).
        model.visual = orig_visual
        wrap = Emb(model, "encode_image").eval()
        print(f"  WARN_NO_REPARAM ({type(e).__name__}: {e}) — latence pessimiste",
              flush=True)

    c32, c8, serve = export_onnx(wrap, pp, name)
    sess = ort.InferenceSession(f"{name}/{serve}", _OPT,
                                providers=["CPUExecutionProvider"])
    del model, wrap
    return {"name": name, "ckpt": hub, "size": list(size), "mean": list(mean),
            "std": list(std), "resample": interp, "resize_mode": rmode,
            "temb": temb, "scale": scale, "bias": bias, "pp": pp, "sess": sess,
            "serve_file": serve, "cos32": c32, "cos8": c8, "dir": name,
            "dim": int(temb.shape[1])}

def load_siglip():
    """La tour SERVIE, telle quelle : même ONNX, même préprocessing (référence)."""
    from huggingface_hub import hf_hub_download
    print("\n=== siglip2 servi (référence) ===", flush=True)
    f = {n: hf_hub_download(SIGLIP_REPO, n, token=HF_TOKEN)
         for n in ("model.fp32.onnx", "meta.json")}
    meta = json.load(open(f["meta.json"]))
    H, W = meta["image_size"]
    pp = make_preprocess(H, W, meta["image_mean"], meta["image_std"],
                         meta["resample"], meta.get("resize_mode", "squash"))
    # les prompts : le repo servi ne les porte plus (le probe P5b a remplacé
    # text_embeds.npy par les prototypes appris) → on ré-encode depuis le socle.
    from transformers import AutoModel, AutoTokenizer
    m = AutoModel.from_pretrained(SIGLIP_CKPT).eval()
    tk = AutoTokenizer.from_pretrained(SIGLIP_CKPT)
    with torch.no_grad():
        t = m.get_text_features(**tk(TEXTS, padding="max_length", max_length=64,
                                     truncation=True, return_tensors="pt"))
    t = t if torch.is_tensor(t) else t.pooler_output
    temb = torch.nn.functional.normalize(t, dim=-1).numpy().astype(np.float32)
    del m
    sess = ort.InferenceSession(f["model.fp32.onnx"], _OPT,
                                providers=["CPUExecutionProvider"])
    return {"name": "siglip2_servi", "ckpt": SIGLIP_CKPT, "size": [H, W],
            "mean": meta["image_mean"], "std": meta["image_std"],
            "resample": meta["resample"], "resize_mode": meta.get("resize_mode", "squash"),
            "temb": temb, "scale": float(meta["logit_scale"]),
            "bias": float(meta["logit_bias"]), "pp": pp, "sess": sess,
            "serve_file": meta.get("serve_file", "model.fp32.onnx"),
            "cos32": meta.get("cos_torch_fp32"), "cos8": meta.get("cos_fp32_int8"),
            "dir": None, "dim": int(temb.shape[1])}

# ══ 2. les trois tours ════════════════════════════════════════════════════════
TOWERS = [load_siglip()] + [load_mc2(n, h) for n, h in MC2_HUBS.items()]

# ══ 3. collecte web — UNE FOIS, partagée par les trois tours ══════════════════
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

def search_urls(cls):
    urls, seen = [], set()
    for q in QUERIES[cls]:
        for attempt in (1, 2):
            try:
                for r in DDGS().images(q, max_results=40):
                    for k in ("image", "thumbnail"):
                        u = r.get(k)
                        if u and u not in seen:
                            seen.add(u); urls.append(u); break
                break
            except Exception as e:
                print(f"  [{cls}] '{q}' tentative {attempt}: {type(e).__name__}")
                time.sleep(20 * attempt)
        time.sleep(2)      # ponytail: politesse anti rate-limit, séquentiel suffit
    return urls

def fetch(url):
    """Renvoie un JPEG canonique ≤512 px : décodable par les 3 tours, RAM bornée."""
    try:
        r = requests.get(url, headers=UA, timeout=8)
        if len(r.content) > 8_000_000:
            return None
        im = Image.open(io.BytesIO(r.content)); im.load()
        if min(im.size) < 120 or max(im.size) / min(im.size) > 4:
            return None
        im = im.convert("RGB")
        im.thumbnail((512, 512))          # même image pour tout le monde
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=92)
        return buf.getvalue()
    except Exception:
        return None

blobs, ylist = [], []
for ci, cls in enumerate(CLASSES):
    urls = search_urls(cls)
    got = 0
    with ThreadPoolExecutor(16) as pool:
        for b in pool.map(fetch, urls[:1400]):
            if b is not None:
                blobs.append(b); ylist.append(ci); got += 1
                if got >= PER_CLASS_CAP:
                    break
    print(f"[{cls}] urls={len(urls)} valides={got}", flush=True)
y_all = np.array(ylist)
assert len(blobs) > 500, f"collecte trop maigre ({len(blobs)}) — ddgs bloqué ?"
print(f"collecte: {len(blobs)} images, {sum(len(b) for b in blobs)/1e6:.0f} Mo RAM",
      flush=True)

# ══ 4. embeddings + latence par tour (fichier SERVI : espace exact du serveur) ══
def embed_all(tw, bs=16):
    out = []
    for i in range(0, len(blobs), bs):
        batch = np.stack([tw["pp"](Image.open(io.BytesIO(b))) for b in blobs[i:i + bs]])
        out.append(tw["sess"].run(None, {"pixel_values": batch})[0])
    return np.concatenate(out).astype(np.float32)   # déjà L2-norm (sortie ONNX)

def latency(tw, n=30):
    x = np.stack([tw["pp"](Image.open(io.BytesIO(blobs[0])))])
    tw["sess"].run(None, {"pixel_values": x})       # warm-up
    t0 = time.time()
    for _ in range(n):
        tw["sess"].run(None, {"pixel_values": x})
    return (time.time() - t0) / n * 1000

for tw in TOWERS:
    t0 = time.time()
    tw["X"] = embed_all(tw)
    tw["ms"] = latency(tw)
    print(f"[{tw['name']}] dim={tw['dim']} embeddings en {(time.time()-t0)/60:.1f} min "
          f"— {tw['ms']:.0f} ms/image (4 threads, {tw['serve_file']})", flush=True)

# dédup décidée sur la tour de RÉFÉRENCE puis appliquée aux trois : même jeu
# d'images pour tout le monde, sinon la comparaison ne veut rien dire.
ref = TOWERS[0]["X"]
keep = np.ones(len(ref), bool)
sims = ref @ ref.T
for i in range(len(ref)):
    if keep[i]:
        dup = (sims[i] > 0.99) & keep
        dup[:i + 1] = False
        keep[dup] = False
del sims
y = y_all[keep]
for tw in TOWERS:
    tw["X"] = tw["X"][keep]
print(f"après dédup: {int(keep.sum())} images", flush=True)

# split stratifié 80/20 — indices PARTAGÉS
tr_idx, te_idx = [], []
for ci in range(len(CLASSES)):
    idx = np.where(y == ci)[0]; np.random.shuffle(idx)
    n_te = max(10, len(idx) // 5) if len(idx) >= MIN_TRAIN else len(idx)
    te_idx += list(idx[:n_te]); tr_idx += list(idx[n_te:])
ytr, yte = y[tr_idx], y[te_idx]
print(f"train={len(tr_idx)} heldout={len(te_idx)}", flush=True)

# ══ 5. protocole P5b v2, IDENTIQUE pour les trois tours ═══════════════════════
def evaluate(Xte, vecs, vec_class, scale, bias, tag):
    logits = scale * (Xte @ vecs.T) + bias
    per = np.full((len(Xte), len(CLASSES)), -1e30, np.float32)
    for j, ci in enumerate(vec_class):
        per[:, ci] = np.maximum(per[:, ci], logits[:, j])
    pred = per.argmax(1)
    recalls, table = [], []
    for ci, c in enumerate(CLASSES):
        m = yte == ci
        if m.sum():
            r = float((pred[m] == ci).mean())
            recalls.append(r); table.append(f"    {c}: {r:.2f} ({int(m.sum())})")
    macro, acc = float(np.mean(recalls)), float((pred == yte).mean())
    corr = per[np.arange(len(pred)), pred][pred == yte]
    med = float(np.median(corr)) if len(corr) else 0.0
    print(f"  {tag}: macro={macro:.4f} acc={acc:.4f} med_logit={med:.2f}", flush=True)
    return macro, acc, table, med

def kmeans_sphere(E, k, iters=30):
    C = E[np.random.choice(len(E), k, replace=False)].copy()
    for _ in range(iters):
        a = (E @ C.T).argmax(1)
        for j in range(k):
            m = E[a == j]
            if len(m):
                v = m.mean(0); C[j] = v / (np.linalg.norm(v) + 1e-9)
    return C

def fit_protos(Xtr):
    protos, pc = [], []
    for ci in range(len(CLASSES)):
        E = Xtr[ytr == ci]
        if len(E) < MIN_TRAIN:
            continue                      # classe maigre : prompts texte seuls
        k = min(4, max(1, len(E) // 50))
        protos.append(kmeans_sphere(E, k)); pc += [ci] * k
    return np.concatenate(protos), pc

def with_text_fallback(vecs, vec_class, tw, covered):
    extra = [(tw["temb"][j], cj) for j, cj in enumerate(PROMPT_CLASS) if cj not in covered]
    if not extra:
        return vecs, list(vec_class)
    return (np.concatenate([vecs, np.stack([e for e, _ in extra])]),
            list(vec_class) + [c for _, c in extra])

def refine(P0, pc0, Xtr, scale, bias, steps=300):
    """BCE sur la sphère, max-pool par classe = math exacte du serveur."""
    Wt = torch.tensor(P0.copy(), requires_grad=True)
    pct = torch.tensor(pc0)
    Xt, Yt = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long)
    targets = torch.zeros(len(Xt), len(CLASSES)).scatter_(1, Yt[:, None], 1.0) * 0.90 + 0.02
    opt = torch.optim.Adam([Wt], lr=5e-4, weight_decay=1e-4)
    for _ in range(steps):
        Wn = Wt / Wt.norm(dim=1, keepdim=True)
        lg = scale * (Xt @ Wn.T) + bias
        per = torch.full((len(Xt), len(CLASSES)), -1e30)
        per = per.scatter_reduce(1, pct[None].expand(len(Xt), -1), lg, reduce="amax")
        loss = torch.nn.functional.binary_cross_entropy_with_logits(per, targets)
        opt.zero_grad(); loss.backward(); opt.step()
    return (Wt / Wt.norm(dim=1, keepdim=True)).detach().numpy().astype(np.float32)

def platt(X, yv, vecs, vec_class, tag):
    """Ajuste (scale, bias) pour que sigmoid(scale·cos+bias) ≈ P(prédiction juste).
    Le serveur lit ces deux nombres dans meta.json — zéro changement de code.
    Init fixe : MobileCLIP2 est CLIP-style (logit_scale ≈ 100, pas de biais) donc
    sa sigmoïde native sature à 1,0 ; le tri (argmax) est invariant, la confiance
    non — et c'est elle qui pilote le seuil 0,35 de l'app."""
    cs = X @ vecs.T
    per = np.full((len(X), len(CLASSES)), -1e30, np.float32)
    for j, ci in enumerate(vec_class):
        per[:, ci] = np.maximum(per[:, ci], cs[:, j])
    pred = per.argmax(1)
    c, ok = per[np.arange(len(pred)), pred], pred == yv
    ct = torch.tensor(c); okt = torch.tensor(ok.astype(np.float32))
    a = torch.tensor(20.0, requires_grad=True); b = torch.tensor(-5.0, requires_grad=True)
    o = torch.optim.Adam([a, b], lr=0.5)
    for _ in range(2000):
        l = torch.nn.functional.binary_cross_entropy_with_logits(a * ct + b, okt)
        o.zero_grad(); l.backward(); o.step()
    av, bv = float(a), float(b)
    conf = 1 / (1 + np.exp(-(av * c + bv)))
    print(f"  [{tag}] platt scale={av:.2f} bias={bv:.2f} "
          f"med_conf_juste={np.median(conf[ok]):.3f} "
          f"med_conf_faux={np.median(conf[~ok]) if (~ok).any() else 0:.3f}", flush=True)
    return av, bv

for tw in TOWERS:
    print(f"\n── {tw['name']} ({tw['dim']}-d, {tw['ms']:.0f} ms/img) ──", flush=True)
    Xtr, Xte = tw["X"][tr_idx], tw["X"][te_idx]
    # calibration d'abord : le raffinement BCE a besoin d'une sigmoïde non saturée
    s, b = platt(Xtr, ytr, tw["temb"], PROMPT_CLASS, tw["name"] + " pré")
    res = {"zeroshot": evaluate(Xte, tw["temb"], PROMPT_CLASS, s, b, "ZEROSHOT")}
    P0, pc0 = fit_protos(Xtr)
    covered = sorted(set(pc0))
    if len(covered) < len(CLASSES):
        print(f"  classes sans prototypes (< {MIN_TRAIN} images) : "
              f"{[CLASSES[i] for i in range(len(CLASSES)) if i not in covered]}")
    pv, pc = with_text_fallback(P0, pc0, tw, covered)
    res["prototypes"] = evaluate(Xte, pv, pc, s, b, "PROTOTYPES_KMEANS")
    P1 = refine(P0, pc0, Xtr, s, b)
    pv1, pc1 = with_text_fallback(P1, pc0, tw, covered)
    res["refined"] = evaluate(Xte, pv1, pc1, s, b, "PROTOTYPES_REFINED")
    uv = np.concatenate([tw["temb"], P1]); uc = list(PROMPT_CLASS) + list(pc0)
    res["union"] = evaluate(Xte, uv, uc, s, b, "UNION_TEXT_PLUS_REFINED")
    win = max(res, key=lambda k: res[k][0])
    tw.update(res=res, win=win, macro=res[win][0], zs=res["zeroshot"][0], pc0=pc0,
              vecs={"zeroshot": (tw["temb"], list(PROMPT_CLASS)),
                    "prototypes": (pv, pc), "refined": (pv1, pc1),
                    "union": (uv, uc)}[win], pre_cal=(s, b))
    print(f"  MEILLEUR={win} macro={tw['macro']:.4f} (zero-shot {tw['zs']:.4f})", flush=True)

# ══ 6. verdict : mêmes images, même protocole, même moteur ════════════════════
siglip, best = TOWERS[0], max(TOWERS[1:], key=lambda t: t["macro"])
d_macro = best["macro"] - siglip["macro"]
speedup = siglip["ms"] / max(best["ms"], 1e-9)
print("\n┌─ comparaison iso (mêmes images héld-out, même protocole) ─")
for tw in TOWERS:
    print(f"│ {tw['name']:<14} macro {tw['macro']:.4f} ({tw['win']:<10}) "
          f"zero-shot {tw['zs']:.4f}  {tw['ms']:6.1f} ms/img  {tw['dim']}-d")
print(f"│ Δmacro {best['name']} − servi = {d_macro:+.4f}   accélération ×{speedup:.1f}")
print("└─", flush=True)
for line in best["res"][best["win"]][2]:
    print(line)
gate = d_macro >= -GATE_MAX_LOSS
print(f"GATE_SWITCH_VISION={'YES' if gate else 'NO'} "
      f"(perte max tolérée {GATE_MAX_LOSS:.2f} pt macro)", flush=True)
# licence : SigLIP2 = Apache-2.0, MobileCLIP2 = apple-amlr (recherche). La
# bascule est donc aussi une décision juridique, pas seulement une mesure.
print("LICENCE: MobileCLIP2 = apple-amlr (recherche) vs SigLIP2 = Apache-2.0 — "
      "à revalider avant commercialisation", flush=True)

# ══ 7. artefacts du meilleur MobileCLIP2 (repo NEUF — la prod ne bouge pas) ═══
fv, fc = best["vecs"]
n_proto = 0 if best["win"] == "zeroshot" else len(best["pc0"])
names, counts = [], {}
for j, cj in enumerate(fc):
    if best["win"] in ("zeroshot", "union") and j < len(TEXTS):
        names.append(TEXTS[j])                       # prompt texte d'origine
    elif best["win"] in ("prototypes", "refined") and j >= n_proto:
        names.append(f"text_fallback:{CLASSES[cj]}")  # classe trop maigre
    else:
        counts[cj] = counts.get(cj, 0) + 1
        names.append(f"learned_prototype:{CLASSES[cj]}:{counts[cj]}")

cal_scale, cal_bias = platt(best["X"][te_idx], yte, fv, fc, "FINAL")
assert cal_scale > 0, "la calibration a inversé le tri — abandon"

out = best["dir"]
np.save(f"{out}/text_embeds.npy", np.ascontiguousarray(fv))
json.dump({"classes": CLASSES, "prompts": names, "prompt_class": [int(c) for c in fc]},
          open(f"{out}/vision_labels.json", "w"), ensure_ascii=False, indent=1)
json.dump({
    "checkpoint": best["ckpt"], "arch": "MobileCLIP2 (FastViT réparamétrisé, open_clip)",
    "license_note": "apple-amlr (recherche) — SigLIP2 servi est Apache-2.0 ; "
                    "revalider la licence avant commercialisation",
    "image_size": best["size"], "image_mean": best["mean"], "image_std": best["std"],
    "resample": best["resample"], "resize_mode": best["resize_mode"],
    "text_max_len": 77, "dim": best["dim"], "opset": 17,
    "logit_scale": cal_scale, "logit_bias": cal_bias,
    "calibration": "Platt sur héld-out web — VISION_BIAS=0 (cuit ici)",
    "cos_torch_fp32": best["cos32"], "cos_fp32_int8": best["cos8"],
    "serve_file": best["serve_file"],
    "probe": "P5c MobileCLIP2 vs SigLIP2 servi, 2026-08-17",
    "probe_winner": best["win"], "probe_macro_heldout": best["macro"],
    "probe_zeroshot_macro": best["zs"],
    "probe_train_images": len(tr_idx), "probe_heldout_images": len(te_idx),
    "ms_per_image_cpu4": best["ms"],
    "vs_siglip2_servi": {"macro": siglip["macro"], "delta_macro": d_macro,
                         "ms_per_image_cpu4": siglip["ms"], "speedup": speedup},
    "switch_recommended": bool(gate),
}, open(f"{out}/meta.json", "w"), indent=1)
json.dump({tw["name"]: {"win": tw["win"], "ms_per_image": tw["ms"], "dim": tw["dim"],
                        "serve_file": tw["serve_file"],
                        "variants": {k: {"macro": v[0], "acc": v[1], "per_class": v[2]}
                                     for k, v in tw["res"].items()}}
           for tw in TOWERS},
          open(f"{out}/probe_report.json", "w"), indent=1, ensure_ascii=False)

from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)
api.create_repo(MC2_REPO, private=True, exist_ok=True)
api.upload_folder(folder_path=out, repo_id=MC2_REPO)
print(f"HF_UPLOAD_DONE={MC2_REPO} ({best['name']}, {sorted(os.listdir(out))})")
if gate:
    print("\n── à mettre dans .env.cloud ET .env.local (jamais le .env généré) ──")
    print(f"VISION_REPO={MC2_REPO}")
    print(f"VISION_MODEL_FILE={best['serve_file']}")
    print("VISION_BIAS=0")
    print(f"# retour arrière : VISION_REPO={SIGLIP_REPO} (intact)")
else:
    print(f"\nSigLIP2 reste servi ({siglip['macro']:.4f} vs {best['macro']:.4f}) ; "
          f"poids MobileCLIP2 gardés sur {MC2_REPO} au cas où.")
