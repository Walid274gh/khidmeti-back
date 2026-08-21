# Khidmeti vision P5b v2 — webly-supervised probe : collecte d'images du web (ddgs),
# embeddings via NOTRE tour image ONNX fp32 (parité serving exacte), prototypes
# sphériques appris par classe, éval held-out vs zero-shot, upload si gate.
# v2 : ~2× images (requêtes ar+fr+en), label smoothing + weight decay contre la
# saturation de confiance (blocage v1), calibration Platt cuite dans
# logit_scale/logit_bias de meta.json (le serveur les lit déjà — zéro changement).
#
# Artefact final = text_embeds.npy/vision_labels.json ENRICHIS (prompts texte
# + prototypes appris) — le serveur ai-vision (max sur les vecteurs d'une
# classe) ne change PAS d'une ligne. Les images brutes ne sont JAMAIS uploadées
# (usage interne d'entraînement uniquement) ; seuls les prototypes (vecteurs
# 768-d) partent sur HF.
#
# Kaggle script kernel CPU, internet ON. Placeholder: {{HF_TOKEN}}
import io, json, os, random, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "ddgs", "onnxruntime", "pillow"], check=True)

import numpy as np
import requests
import torch
from PIL import Image

SEED = 20260804
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

HF_TOKEN = "{{HF_TOKEN}}"
REPO = "Walidrbh27/khidmeti-vision"
PER_CLASS_CAP = 400          # images valides max / classe (v2: 220→400)
MIN_TRAIN = 40               # en-dessous : la classe garde ses prompts texte seuls
GATE_MACRO_GAIN = 0.05       # le gagnant doit battre le zero-shot de ≥5 pts macro
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36"}

# ── 1. artefacts de prod (parité totale : même ONNX, même preprocessing) ─────
from huggingface_hub import hf_hub_download, HfApi
files = {f: hf_hub_download(REPO, f, token=HF_TOKEN)
         for f in ("model.fp32.onnx", "text_embeds.npy", "vision_labels.json", "meta.json")}
meta = json.load(open(files["meta.json"]))
labels = json.load(open(files["vision_labels.json"]))
temb = np.load(files["text_embeds.npy"]).astype(np.float32)
CLASSES = labels["classes"]
SCALE, BIAS = float(meta["logit_scale"]), float(meta["logit_bias"])
H, W = meta["image_size"]
MEAN = np.array(meta["image_mean"], dtype=np.float32)
STD = np.array(meta["image_std"], dtype=np.float32)
RESAMPLE = getattr(Image, meta["resample"].upper())
print(f"classes={len(CLASSES)} text_prompts={temb.shape[0]} scale={SCALE:.2f} bias={BIAS:.2f}")

import onnxruntime as ort
sess = ort.InferenceSession(files["model.fp32.onnx"], providers=["CPUExecutionProvider"])

def preprocess(img):
    img = img.convert("RGB").resize((W, H), RESAMPLE)
    x = np.asarray(img, dtype=np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1)

def embed(imgs, bs=16):
    out = []
    for i in range(0, len(imgs), bs):
        batch = np.stack([preprocess(im) for im in imgs[i:i+bs]])
        out.append(sess.run(None, {"pixel_values": batch})[0])
    return np.concatenate(out).astype(np.float32)   # déjà L2-norm (sortie ONNX)

# ── 2. requêtes web par métier — photos de PANNES (leçon P1), ar+fr+en (v2) ──
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
                         "réparation électroménager technicien", "machine à laver démontée réparation"],
    "mover": ["cartons déménagement empilés", "camion déménagement meubles",
              "meubles emballés plastique déménagement", "moving boxes stacked room",
              "moving truck loading furniture", "déménagement appartement cartons",
              "نقل أثاث شاحنة", "عفش نقل كراتين",
              "monte-meuble déménagement", "furniture wrapped blankets moving"],
    "mechanic": ["moteur voiture en panne", "voiture capot ouvert panne", "pneu crevé voiture",
                 "car engine bay problem", "flat tire car", "voiture en panne bord route",
                 "tableau de bord voyant moteur allumé", "سيارة معطلة محرك", "ميكانيكي سيارات ورشة",
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
    "caterer": ["couscous plat traditionnel", "traiteur buffet mariage", "gâteaux traditionnels algériens",
                "catering trays food party", "wedding buffet table food", "plats traiteur fête",
                "طعام تقليدي جزائري", "حلويات عيد تقليدية",
                "plats mariage traiteur oriental", "couscous garni viande légumes"],
    "none": ["selfie personne visage", "screenshot application téléphone", "document papier texte scan",
             "rue ville paysage urbain", "chat chien animal compagnie", "jardin pelouse arbres",
             "salon propre moderne intérieur", "montagne plage paysage nature",
             "groupe personnes réunion photo", "صورة شخصية سيلفي", "شاشة هاتف تطبيق",
             "facture papier administratif"],
}
assert set(QUERIES) == set(CLASSES), "requêtes ≠ classes de prod"

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
                            seen.add(u)
                            urls.append(u)
                            break
                break
            except Exception as e:
                print(f"  [{cls}] '{q}' tentative {attempt}: {type(e).__name__}")
                time.sleep(20 * attempt)
        time.sleep(2)   # ponytail: politesse anti rate-limit, séquentiel suffit
    return urls

def fetch(url):
    try:
        r = requests.get(url, headers=UA, timeout=8)
        if len(r.content) > 8_000_000:
            return None
        im = Image.open(io.BytesIO(r.content))
        im.load()
        if min(im.size) < 120 or max(im.size) / min(im.size) > 4:
            return None
        return im.convert("RGB")
    except Exception:
        return None

col_embs, col_lbls = [], []
for ci, cls in enumerate(CLASSES):
    urls = search_urls(cls)
    imgs = []
    with ThreadPoolExecutor(16) as pool:
        for im in pool.map(fetch, urls[:1400]):
            if im is not None:
                imgs.append(im)
                if len(imgs) >= PER_CLASS_CAP:
                    break
    if not imgs:
        print(f"[{cls}] 0 image — classe sans prototypes")
        continue
    e = embed(imgs)
    col_embs.append(e); col_lbls += [ci] * len(e)
    print(f"[{cls}] urls={len(urls)} valides={len(imgs)}")

X = np.concatenate(col_embs)
y = np.array(col_lbls)

# ── 3. dédup global (miniatures/reposts) : cos>0.99 → garde le premier ───────
keep = np.ones(len(X), bool)
sims = X @ X.T
for i in range(len(X)):
    if keep[i]:
        dup = (sims[i] > 0.99) & keep
        dup[:i + 1] = False
        keep[dup] = False
X, y = X[keep], y[keep]
print(f"après dédup: {len(X)} images")

# ── 4. split stratifié 80/20 ─────────────────────────────────────────────────
tr_idx, te_idx = [], []
for ci in range(len(CLASSES)):
    idx = np.where(y == ci)[0]; np.random.shuffle(idx)
    n_te = max(10, len(idx) // 5) if len(idx) >= MIN_TRAIN else len(idx)
    te_idx += list(idx[:n_te]); tr_idx += list(idx[n_te:])
Xtr, ytr, Xte, yte = X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]
print(f"train={len(Xtr)} heldout={len(Xte)}")

def evaluate(vecs, vec_class, tag):
    logits = SCALE * (Xte @ vecs.T) + BIAS
    per_cls = np.full((len(Xte), len(CLASSES)), -1e30, np.float32)
    for j, ci in enumerate(vec_class):
        per_cls[:, ci] = np.maximum(per_cls[:, ci], logits[:, j])
    pred = per_cls.argmax(1)
    recalls, table = [], []
    for ci, c in enumerate(CLASSES):
        m = yte == ci
        if m.sum():
            r = float((pred[m] == ci).mean())
            recalls.append(r); table.append(f"  {c}: {r:.2f} ({int(m.sum())})")
    macro = float(np.mean(recalls)); acc = float((pred == yte).mean())
    print(f"{tag}: macro={macro:.4f} acc={acc:.4f}")
    correct = per_cls[np.arange(len(pred)), pred][pred == yte]
    med = float(np.median(correct)) if len(correct) else 0.0
    print(f"  médiane logit corrects={med:.2f} → sigmoïde={1/(1+np.exp(-med)):.3f}")
    return macro, acc, table, med

results = {}
results["zeroshot"] = evaluate(temb, labels["prompt_class"], "ZEROSHOT")

# prototypes: k-means sphérique par classe (k selon la taille), puis raffinement
def kmeans_sphere(E, k, iters=30):
    C = E[np.random.choice(len(E), k, replace=False)]
    for _ in range(iters):
        a = (E @ C.T).argmax(1)
        for j in range(k):
            m = E[a == j]
            if len(m):
                v = m.mean(0); C[j] = v / (np.linalg.norm(v) + 1e-9)
    return C

protos, proto_class = [], []
for ci in range(len(CLASSES)):
    E = Xtr[ytr == ci]
    if len(E) < MIN_TRAIN:
        print(f"[{CLASSES[ci]}] {len(E)} < {MIN_TRAIN} — prompts texte seulement")
        continue
    k = min(4, max(1, len(E) // 50))   # v2: plus d'images → jusqu'à 4 protos/classe
    protos.append(kmeans_sphere(E, k)); proto_class += [ci] * k
P0 = np.concatenate(protos)
covered = sorted(set(proto_class))

def with_text_fallback(vecs, vec_class):
    """classes sans prototypes → leurs prompts texte restent dans le jeu."""
    extra = [(temb[j], cj) for j, cj in enumerate(labels["prompt_class"]) if cj not in covered]
    if not extra:
        return vecs, list(vec_class)
    ev = np.stack([e for e, _ in extra])
    return np.concatenate([vecs, ev]), list(vec_class) + [c for _, c in extra]

pv, pc = with_text_fallback(P0, proto_class)
results["prototypes"] = evaluate(pv, pc, "PROTOTYPES_KMEANS")

# raffinement BCE sur la sphère (max-pool par classe = même math que serving)
Wt = torch.tensor(P0.copy(), requires_grad=True)
pc_t = torch.tensor(proto_class)
Xt, Yt = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long)
# v2 : cibles lissées (0.92/0.02) + weight decay — la BCE dure de v1 saturait
# les confiances à 1.0, ce qui aurait détruit le routage d'incertitude 0.35
onehot = torch.zeros(len(Xt), len(CLASSES)).scatter_(1, Yt[:, None], 1.0)
targets = onehot * 0.90 + 0.02
opt = torch.optim.Adam([Wt], lr=5e-4, weight_decay=1e-4)
for step in range(300):
    Wn = Wt / Wt.norm(dim=1, keepdim=True)
    lg = SCALE * (Xt @ Wn.T) + BIAS
    per = torch.full((len(Xt), len(CLASSES)), -1e30)
    per = per.scatter_reduce(1, pc_t[None].expand(len(Xt), -1), lg, reduce="amax")
    loss = torch.nn.functional.binary_cross_entropy_with_logits(per, targets)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 99:
        print(f"  refine step {step+1} loss={loss.item():.4f}")
P1 = (Wt / Wt.norm(dim=1, keepdim=True)).detach().numpy().astype(np.float32)
pv, pc = with_text_fallback(P1, proto_class)
results["refined"] = evaluate(pv, pc, "PROTOTYPES_REFINED")

uv = np.concatenate([temb, P1])
uc = list(labels["prompt_class"]) + list(proto_class)
results["union"] = evaluate(uv, uc, "UNION_TEXT_PLUS_REFINED")

# ── 5. calibration Platt (v2) ────────────────────────────────────────────────
# La confiance servie = sigmoid(scale·cos + bias) ; le serveur lit scale/bias
# dans meta.json. On les ré-ajuste pour que la confiance ≈ P(prédiction juste),
# mesurée sur le heldout. argmax inchangé (scale>0) → aucune régression de tri.
def cos_top1(vecs, vec_class):
    cs = Xte @ vecs.T
    per = np.full((len(Xte), len(CLASSES)), -1e30, np.float32)
    for j, ci in enumerate(vec_class):
        per[:, ci] = np.maximum(per[:, ci], cs[:, j])
    pred = per.argmax(1)
    return per[np.arange(len(pred)), pred], pred == yte

def platt(vecs, vec_class, tag):
    c, ok = cos_top1(vecs, vec_class)
    ct, okt = torch.tensor(c), torch.tensor(ok.astype(np.float32))
    a = torch.tensor(float(SCALE), requires_grad=True)
    b = torch.tensor(float(BIAS), requires_grad=True)
    o = torch.optim.Adam([a, b], lr=0.5)
    for _ in range(2000):
        l = torch.nn.functional.binary_cross_entropy_with_logits(a * ct + b, okt)
        o.zero_grad(); l.backward(); o.step()
    av, bv = float(a), float(b)
    conf = 1 / (1 + np.exp(-(av * c + bv)))
    print(f"[{tag}] platt scale={av:.2f} bias={bv:.2f} "
          f"med_conf_correct={np.median(conf[ok]):.3f} "
          f"med_conf_wrong={np.median(conf[~ok]) if (~ok).any() else 0:.3f}")
    return av, bv

# ── 6. gate + upload ─────────────────────────────────────────────────────────
base_macro = results["zeroshot"][0]
winner = max(("prototypes", "refined", "union"), key=lambda k: results[k][0])
gain = results[winner][0] - base_macro
print(f"WINNER={winner} macro={results[winner][0]:.4f} zero-shot={base_macro:.4f} gain={gain:+.4f}")
for line in results[winner][2]:
    print(line)
print("GATE_PLUS5_PASS=" + ("YES" if gain >= GATE_MACRO_GAIN else "NO"))
if gain < GATE_MACRO_GAIN:
    # info décision : la calibration seule (zero-shot Platt) reste possible à la main
    platt(temb, labels["prompt_class"], "RECO_ZEROSHOT_PLATT")
    print("GATE_FAILED_NO_UPLOAD — zero-shot reste servi")
    sys.exit(0)

final_v, final_c = {"prototypes": with_text_fallback(P0, proto_class),
                    "refined":    with_text_fallback(P1, proto_class),
                    "union":      (uv, uc)}[winner]
names = []
counts = {}
for j, cj in enumerate(final_c):
    if winner == "union" and j < len(labels["prompts"]):
        names.append(labels["prompts"][j])
    else:
        counts[cj] = counts.get(cj, 0) + 1
        names.append(f"learned_prototype:{CLASSES[cj]}:{counts[cj]}")

cal_scale, cal_bias = platt(final_v, final_c, "WINNER_PLATT")
assert cal_scale > 0, "calibration a inversé le tri — abandon"

np.save("text_embeds.npy", np.ascontiguousarray(final_v))
json.dump({"classes": CLASSES, "prompts": names, "prompt_class": [int(c) for c in final_c]},
          open("vision_labels.json", "w"), ensure_ascii=False, indent=1)
meta.update({"probe": "P5b v2 webly-supervised 2026-08-04", "probe_winner": winner,
             "probe_macro_heldout": results[winner][0], "probe_zeroshot_macro": base_macro,
             "probe_train_images": int(len(Xtr)), "probe_heldout_images": int(len(Xte)),
             "probe_median_correct_logit": results[winner][3],
             "logit_scale": cal_scale, "logit_bias": cal_bias,
             "calibration": "Platt web-heldout — remettre VISION_BIAS=0 (cuit ici)"})
json.dump(meta, open("meta.json", "w"), indent=1)
json.dump({k: {"macro": v[0], "acc": v[1], "per_class": v[2]} for k, v in results.items()},
          open("probe_report.json", "w"), indent=1)

api = HfApi(token=HF_TOKEN)
for f in ("text_embeds.npy", "vision_labels.json", "meta.json", "probe_report.json"):
    api.upload_file(path_or_fileobj=f, path_in_repo=f, repo_id=REPO)
    print("uploaded", f)
print("ALL DONE —", REPO)  # confiance calibrée cuite dans meta — VISION_BIAS=0 côté serveur
