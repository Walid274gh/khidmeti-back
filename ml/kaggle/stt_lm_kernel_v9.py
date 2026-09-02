#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
# ml/kaggle/stt_lm_kernel_v9.py — v9a : on ne réentraîne RIEN, on répare le DÉCODAGE.
#
# Constat v8, mêmes 831 clips Casablanca-Algeria, même moteur int8 CPU 4 threads :
#   whisper medium+LoRA servi ..... WER 0,6300  CER 0,2373  4,181×RT
#   wav2vec2 darija CTC v8 ........ WER 0,6892  CER 0,2508  0,182×RT   (×22,9)
# Le CER est à parité (+1,35 pt) pendant que le WER perd 5,92 pts : les
# CARACTÈRES sont bons, les MOTS sont faux. C'est la signature exacte d'un CTC
# décodé en argmax sans modèle de langue — whisper, lui, a un LM dans son
# décodeur auto-régressif, c'est de là que vient tout son avantage.
#
# Donc v9a : mêmes poids, même encodeur, on ajoute la pièce qui manque —
# n-gram darija algérien + recherche par faisceau à fusion superficielle.
#   • AUCUN GPU (pousser avec CPU=1) : on rejoue les logits du moteur SERVI.
#   • Le WER mesuré ici n'est plus une estimation : c'est model.int8.onnx sur
#     les 831 clips, donc directement comparable au 0,6300 de whisper.
#   • Poids inchangés → si le gate passe, on sert le CTC sans réentraînement.
#
# GATE = WER(faisceau+LM, int8, 831 clips) < 0,6097 (whisper apparié prod ;
# l'ancien 0,6300 venait d'une estimation sans vad+prompt — trop laxiste de ~2 pts).
#   passe  → int8 + vocab + lm.npz + meta sur le repo servi, lignes .env imprimées
#   échoue → lm.npz + meta sur le repo candidat, et on garde whisper servi.
#
# Le décodage (greedy ET faisceau) est écrit ICI et rejoué à l'identique dans
# docker/ai-stt/server.py : les asserts de demo() sont la parité entre les deux.
# ══════════════════════════════════════════════════════════════════════════════
import bisect, gc, hashlib, json, math, os, random, re, subprocess, sys, time

T0 = time.time()
DEADLINE_S = 8.0 * 3600          # ponytail: budget mur ; au-delà on publie ce qu'on a

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "datasets==3.6.0", "soundfile", "librosa", "jiwer",
                "onnxruntime", "huggingface_hub"], check=True)

import numpy as np
import jiwer
from datasets import load_dataset, Audio
from huggingface_hub import hf_hub_download, HfApi
import onnxruntime as ort

SEED = 20260817                  # IDENTIQUE v8 → même découpe val/train DZ
random.seed(SEED); np.random.seed(SEED)

HF_TOKEN = "{{HF_TOKEN}}"
os.environ["HF_TOKEN"] = HF_TOKEN
CAND_REPO = "Walidrbh27/khidmeti-stt-ctc-cand"   # poids courants (24 h, B_ep8; le gate échoue → ici)
REPO      = "Walidrbh27/khidmeti-stt-ctc"        # repo servi si le gate passe
# Constante du gate MISE À JOUR (run paired khidmeti-stt-paired) : whisper
# servi en réglages PRODUCTION (language=ar beam=1 vad=True prompt=True,
# int8 4 threads, 831 clips) = 0.6097, IC95 [0.5930 ; 0.6276] — l'ancienne
# constante 0.6300 venait d'une estimation sans vad+prompt et était trop
# laxiste de ~2 pts : un CTC à 0.62 aurait été servi "moins bien que whisper
# réel". CER whisper mesuré au même run = 0.2368.
SERVED_WER, SERVED_CER = 0.6097, 0.2368          # whisper servi (paired, prod)
GREEDY_V8_EST = 0.6064                           # greedy int8 MESURÉ (25/08, log S5 :
                                                 # SUB_ONNX_INT8_WER) sur les poids -cand
                                                 # actuels — si le mesuré s'écarte de >3 pts,
                                                 # ce ne sont pas les bons logits

LM_ORDER, LM_MINCOUNT, LM_WORDS = 3, 2, 12_000_000
LM_ORDERS = [3, 4]                # build_lm coûte 0,5 min : l'ordre se mesure
# Grilles resserrées par DEUX runs de mesures, pas par intuition :
#   alpha 0.7 s'effondre (0.89 au run 1) et 0.5 aussi → retirés, on garde 0.45
#     comme témoin de bord haut ; l'optimum mesuré est 0.2 aux deux runs.
#   beta optimum 1.0, loin du bord (3.5/5.0 jamais gagnants) → range réduit.
#   floor : mécanisme PROUVÉ en assert mais mesuré à ZÉRO effet deux fois de
#     suite (l'optimum est −30 = désactivé) → figé, plus balayé. Le budget de
#     grille ainsi libéré part dans le faisceau, qui lui était collé au bord.
ALPHAS = [0.1, 0.2, 0.3, 0.45]
BETAS  = [0.0, 1.0, 2.0]
FLOORS = [-30.0]
# beam 32 était le BORD de la grille et progressait ENCORE (8→0.6204, 16→0.6110,
# 32→0.6052). On a ×14 de marge de vitesse sur whisper : le faisceau est le
# levier le moins cher qui reste.
BEAMS  = [32, 64, 128]
BEAM0, PRUNE = 16, 1e-4
# Interpolation log-linéaire écrit×parlé : w = poids du LM ÉCRIT (w=1.0 → écrit
# seul, le témoin). Le run précédent a « testé » le corpus parlé en le
# CONCATÉNANT au corpus écrit : 55 573 lignes contre 12 M de mots = 1,7 % des
# comptes, donc un non-test. Ici les deux LM sont séparés et pondérés.
MIX_WEIGHTS = [1.0, 0.85, 0.7, 0.5]
THREADS = 4                                      # = STT_THREADS en prod

# ---- normalisation arabe pour WER (IDENTIQUE v3/v6/v7/v8 — comparabilité) ----
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

# ══ DÉCODEUR — source de vérité, copie identique dans docker/ai-stt/server.py ══
# Stdlib seule : kenlm n'a aucune wheel (tar.gz, compile boost+cmake) et
# pyctcdecode sans kenlm ne fait rien de plus que l'argmax. demo() ci-dessous est
# la parité entre cette copie et celle du serveur.
NEG = -1e30
TOPK = 8          # plafond de sécurité : avec prune=1e-4 il y a 1 à 3 candidats


def log_softmax(rows):
    """Logits bruts (liste de listes) → log-probabilités."""
    out = []
    for r in rows:
        m = max(r)
        s = math.log(sum(math.exp(x - m) for x in r))
        out.append([x - m - s for x in r])
    return out


def ctc_greedy(ids, id2tok, blank, special):
    out, prev = [], -1
    for i in ids:
        if i != prev and i != blank:
            t = id2tok[i]
            if t not in special:
                out.append(t)
        prev = i
    return " ".join("".join(out).replace("|", " ").split())


def _h(s):
    # 63 bits, pas 64 : les clés doivent tenir dans un int64 signé, sinon un
    # tableau numpy uint64 comparé à un int Python peut passer par float64 et
    # bisect renvoie silencieusement la mauvaise entrée.
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(),
                          "little") & 0x7FFFFFFFFFFFFFFF


def build_lm(lines, order=3, min_count=2):
    """n-gram à backoff bête (Brants et al. 2007), stocké en (hachages triés,
    log-scores) : bisect marche sur une liste comme sur un tableau numpy, donc le
    serveur charge un .npz sans changer une ligne d'ici."""
    # ponytail: compte avec des clés chaînes (≈3 Go de pic pour 900 k lignes) —
    # si la RAM manque, baisser LM_WORDS ou compter par tranches.
    c, n_tok = {}, 0
    for ln in lines:
        w = ln.split()
        n_tok += len(w)
        for n in range(1, order + 1):
            for i in range(len(w) - n + 1):
                s = " ".join(w[i:i + n])
                c[s] = c.get(s, 0) + 1
    assert n_tok, "corpus vide"

    keys, logp = [], []
    for s, v in c.items():
        if " " in s:                        # n-gramme d'ordre ≥ 2
            if v < min_count:
                continue
            d = c.get(s.rsplit(" ", 1)[0])  # c(histoire) = les n-1 premiers mots
            if not d:
                continue
            logp.append(math.log(v / d))
        else:                               # unigramme : gardé même à 1, c'est le plancher
            logp.append(math.log(v / n_tok))
        keys.append(_h(s))

    o = sorted(range(len(keys)), key=keys.__getitem__)
    return {"keys": [keys[i] for i in o], "logp": [logp[i] for i in o],
            "order": order, "backoff": math.log(0.4),
            "oov": math.log(1.0 / (n_tok + 1)), "n_tok": n_tok}


def lm_logp(lm, hist, word):
    # Interpolation LOG-LINÉAIRE de deux LM : mix = (lm_a, lm_b, poids_de_a).
    # Log-linéaire et non linéaire-en-probas : en fusion superficielle le score
    # est déjà additif, donc deux termes pondérés se comportent comme deux
    # fusions indépendantes — et ça reste exact au aller-retour .npz (chaque LM
    # garde son propre tableau trié).
    mx = lm.get("mix")
    if mx:
        a, b, w = mx
        return w * lm_logp(a, hist, word) + (1.0 - w) * lm_logp(b, hist, word)
    keys, lp = lm["keys"], lm["logp"]
    step, bo = float(lm["backoff"]), 0.0
    for n in range(min(len(hist), int(lm["order"]) - 1), -1, -1):
        k = _h(" ".join(tuple(hist)[len(hist) - n:] + (word,)) if n else word)
        j = bisect.bisect_left(keys, k)
        if j < len(keys) and keys[j] == k:
            return float(lp[j]) + bo
        bo += step                          # 0,4 par niveau de repli
    return float(lm["oov"]) + bo


def _lae(a, b):
    if a < b:
        a, b = b, a
    return a if b <= a - 30 else a + math.log1p(math.exp(b - a))


def beam_decode(rows, id2tok, blank, special, lm=None, alpha=0.0, beta=0.0,
                beam=16, prune=1e-4, delim="|", floor=-30.0):
    """rows = sortie de log_softmax. État = (mots finis, mot en cours) : c'est
    exactement le texte rendu, donc deux alignements du même texte fusionnent.
    floor = plancher du score LM par mot. Sans lui, un mot juste mais absent du
    corpus coûte log(1/N) ≈ −16 nats, et COLLER deux mots ne paie qu'un seul
    malus au lieu de deux : le LM soude les mots, ce qui est la pire faute
    possible pour le WER (deux erreurs par soudure). Observé au run v9a
    («صعا نقولك» → «صعانقولك»)."""
    lprune = math.log(prune)
    # Les spéciaux ne s'écrivent pas ET coupent les répétitions, exactement comme
    # le blank chez ctc_greedy (i != prev) → on les additionne au blank.
    mute = {i for i, t in enumerate(id2tok) if i == blank or t in special}
    order = int(lm["order"]) if lm else 1
    hist_n = order - 1

    def bonus(words, word):
        if lm is None or not word:
            return 0.0
        return alpha * max(lm_logp(lm, words[-hist_n:] if hist_n else (), word),
                           floor) + beta

    beams = {((), ""): [0.0, NEG, 0.0]}     # état → [p_blank, p_non_blank, lm]
    for row in rows:
        pmute = NEG
        for i in mute:
            pmute = _lae(pmute, row[i])
        cand = [(row[i], i) for i in range(len(row))
                if row[i] > lprune and i not in mute]
        if len(cand) > TOPK:
            cand.sort(reverse=True)
            del cand[TOPK:]

        nxt = {}
        for st, (pb, pnb, lms) in beams.items():
            tot = _lae(pb, pnb)
            e = nxt.get(st)
            if e is None:
                e = nxt[st] = [NEG, NEG, lms]
            e[0] = _lae(e[0], tot + pmute)          # blank/spécial : état inchangé

            words, cur = st
            last = cur[-1] if cur else (delim if words else "")
            for p, i in cand:
                tok = id2tok[i]
                if tok == last:
                    e[1] = _lae(e[1], pnb + p)      # répétition collée : fusionnée
                    src = pb                        # séparée par un blank : elle sort
                else:
                    src = tot
                if tok == delim:
                    if cur:
                        ns, bo = (words + (cur,), ""), bonus(words, cur)
                    else:
                        ns, bo = st, 0.0            # espaces multiples : rien à écrire
                else:
                    ns, bo = (words, cur + tok), 0.0
                if ns == st:
                    e[1] = _lae(e[1], src + p)
                else:
                    f = nxt.get(ns)
                    if f is None:                   # lms est fonction de l'état seul
                        f = nxt[ns] = [NEG, NEG, lms + bo]
                    f[1] = _lae(f[1], src + p)

        if len(nxt) > beam:
            keep = sorted(nxt.items(),
                          key=lambda kv: -(_lae(kv[1][0], kv[1][1]) + kv[1][2]))[:beam]
            nxt = dict(keep)
        beams = nxt

    best, bestsc = "", -1e300
    for (words, cur), (pb, pnb, lms) in beams.items():
        sc = _lae(pb, pnb) + lms + bonus(words, cur)   # dernier mot partiel compris
        if sc > bestsc:
            bestsc, best = sc, " ".join(words + ((cur,) if cur else ()))
    return best


def demo():
    id2tok = ["<pad>", "|", "ا", "ب", "ت", "<s>"]
    blank, special = 0, {"<pad>", "<s>"}
    kw = dict(id2tok=id2tok, blank=blank, special=special)

    def peaky(seq):
        rows = []
        for i in seq:
            r = [-12.0] * len(id2tok)
            r[i] = 0.0
            rows.append(r)
        return log_softmax(rows)

    # 1. sans LM et sur des logits piqués, le faisceau DOIT retomber sur l'argmax :
    #    c'est le test qui attrape une récurrence p_blank/p_non_blank cassée.
    rnd = random.Random(7)
    for _ in range(60):
        seq = [rnd.randrange(len(id2tok)) for _ in range(rnd.randint(1, 14))]
        g = ctc_greedy(seq, id2tok, blank, special)
        b = beam_decode(peaky(seq), **kw)
        assert g == b, (seq, g, b)

    # 2. règle de répétition + parité des tokens spéciaux
    assert beam_decode(peaky([2, 0, 2]), **kw) == "اا"      # séparée par blank
    assert beam_decode(peaky([2, 2]), **kw) == "ا"          # collée : fusionnée
    assert beam_decode(peaky([2, 5, 2]), **kw) == "اا"      # <s> coupe comme blank
    assert beam_decode(peaky([1, 2, 1, 1, 3, 1]), **kw) == "ا ب"   # délims en trop
    assert beam_decode(peaky([0, 0]), **kw) == ""           # silence

    # 3. un LM doit renverser un décodage acoustiquement serré
    lm = build_lm(["ابت بات", "ابت بات", "ابت بات", "ابت تاب"],
                  order=3, min_count=1)
    amb = peaky([2, 3, 4, 1]) + log_softmax([
        [-12, -12, -12, -0.3, 0.0, -12],    # ت légèrement devant ب
        [-12, -12, 0.0, -12, -12, -12],     # ا
        [-12, -12, -12, 0.0, -0.3, -12],    # ب légèrement devant ت
    ])
    assert beam_decode(amb, **kw) == "ابت تاب"                       # acoustique seule
    assert beam_decode(amb, lm=lm, alpha=1.0, **kw) == "ابت بات"     # le LM tranche

    # 4. le LM lui-même : vu > rare > hors vocabulaire, et rien d'infini
    seen = lm_logp(lm, ("ابت",), "بات")
    rare = lm_logp(lm, ("ابت",), "تاب")
    oov = lm_logp(lm, ("ابت",), "زززز")
    assert seen > rare > oov, (seen, rare, oov)
    assert math.isfinite(oov) and oov < 0
    assert abs(seen - math.log(3 / 4)) < 1e-9, seen   # c(ابت بات)/c(ابت) = 3/4
    assert abs(rare - math.log(1 / 4)) < 1e-9, rare
    assert len(lm["keys"]) == len(lm["logp"]) == len(set(lm["keys"]))
    assert lm["keys"] == sorted(lm["keys"])          # bisect exige le tri

    # 5. beta pousse à découper en mots, alpha=0 le laisse tranquille
    assert beam_decode(amb, lm=lm, alpha=0.0, beta=0.0, **kw) == "ابت تاب"

    # 6. le plancher borne le malus d'un mot inconnu. Sans lui, SOUDER deux mots
    #    inconnus ne paie qu'UN malus au lieu de deux : le LM soude, et une
    #    soudure = deux fautes de WER. Split gagne ssi beta > alpha·|plancher|.
    big = build_lm(["ا ب ت"] * 1000, order=3, min_count=1)   # n_tok élevé → oov ≈ −8
    soft = (peaky([4, 2, 3])                                 # "تاب"
            + log_softmax([[-0.7, -0.7, -12, -12, -12, -12]])  # blank ou délimiteur
            + peaky([4, 2, 3]))                              # "تاب"
    assert lm_logp(big, (), "تاب") < -7, lm_logp(big, (), "تاب")
    assert " " not in beam_decode(soft, lm=big, alpha=1.0, beta=4.0, **kw)
    assert " " in beam_decode(soft, lm=big, alpha=1.0, beta=4.0, floor=-3.0, **kw)

    # 7. interpolation log-linéaire écrit×parlé : w=1 → LM a seul, w=0 → LM b
    #    seul, entre les deux ça interpole VRAIMENT. Le test qui compte : deux LM
    #    qui se CONTREDISENT doivent renverser le décodage selon w — sinon le
    #    mélange serait un no-op silencieux (exactement le piège du run
    #    précédent, où le corpus parlé concaténé pesait 1,7 % des comptes).
    lm_a = build_lm(["ابت بات"] * 3 + ["ابت تاب"], order=3, min_count=1)
    lm_b = build_lm(["ابت تاب"] * 3 + ["ابت بات"], order=3, min_count=1)   # avis inverse
    def mix(w):
        return {"order": 3, "mix": (lm_a, lm_b, w)}
    for hh, ww in ((("ابت",), "بات"), ((), "تاب"), (("ابت",), "زززز")):
        a_, b_ = lm_logp(lm_a, hh, ww), lm_logp(lm_b, hh, ww)
        assert abs(lm_logp(mix(1.0), hh, ww) - a_) < 1e-12, (hh, ww)
        assert abs(lm_logp(mix(0.0), hh, ww) - b_) < 1e-12, (hh, ww)
        assert abs(lm_logp(mix(0.5), hh, ww) - 0.5 * (a_ + b_)) < 1e-12, (hh, ww)
    assert beam_decode(amb, lm=mix(1.0), alpha=1.0, **kw) == "ابت بات"    # a tranche
    assert beam_decode(amb, lm=mix(0.0), alpha=1.0, **kw) == "ابت تاب"    # b tranche
# ══ fin décodeur ══════════════════════════════════════════════════════════════

demo()
print("DECODER_SELFCHECK_OK", flush=True)

# ══ 1. données : Casablanca:Algeria, découpe REJOUÉE de v8 ════════════════════
def clean(t):
    return norm(re.sub(r"<[^>]*>|\[[^\]]*\]", " ", t))

print("Loading Casablanca:Algeria …", flush=True)
ds = load_dataset("UBC-NLP/Casablanca", "Algeria")
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

def cas_items(split):
    out = []
    for ex in ds[split]:
        t, d = clean(ex["transcription"]), float(ex["duration"])
        if 0.5 < d < 29.5 and len(t) > 1:
            out.append({"text": t, "dur": d,
                        "array": np.asarray(ex["audio"]["array"], dtype=np.float32)})
    return out

tr_full = cas_items("validation")      # première consommation d'aléa = découpe v8
test    = cas_items("test")
random.shuffle(tr_full)
val = tr_full[:max(1, len(tr_full) // 10)]
del tr_full; gc.collect()
assert len(test) == 831, f"test différent des runs v3/v7/v8 : {len(test)}"
print(f"val={len(val)} test={len(test)} "
      f"({sum(i['dur'] for i in test)/3600:.2f} h de test)", flush=True)

# ══ 2. moteur servi : model.int8.onnx du repo candidat v8 ═════════════════════
def pull(name, repo=CAND_REPO):
    return hf_hub_download(repo, name, token=HF_TOKEN)

meta_v8 = json.load(open(pull("meta.json"), encoding="utf-8"))
vocab   = json.load(open(pull("ctc_vocab.json"), encoding="utf-8"))
ID2TOK, BLANK = vocab["id2tok"], vocab["blank_id"]
DELIM, SPECIAL = vocab["delim"], set(vocab.get("special", []))
SERVE_FILE = meta_v8.get("serve_file", "model.int8.onnx")
print(f"moteur={SERVE_FILE} vocab={len(ID2TOK)} delim={DELIM!r} blank={BLANK}", flush=True)

_OPT = ort.SessionOptions(); _OPT.intra_op_num_threads = THREADS
sess = ort.InferenceSession(pull(SERVE_FILE), _OPT, providers=["CPUExecutionProvider"])

def logits_of(items, tag):
    """Un seul passage encodeur : ensuite tous les décodages sont gratuits."""
    out, t0 = [], time.time()
    for k, it in enumerate(items):
        x = it["array"]
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)      # do_normalize, cf. server.py
        out.append(sess.run(None, {"input_values": x[None]})[0][0].astype(np.float16))
        if k and k % 200 == 0:
            print(f"  [{tag}] {k}/{len(items)} — {left()/3600:.1f} h restantes", flush=True)
    rt = (time.time() - t0) / max(1e-9, sum(i["dur"] for i in items))
    print(f"{tag}_ENCODER_xRT={rt:.4f} frames_tot={sum(len(l) for l in out)}", flush=True)
    return out, rt

val_lg, _        = logits_of(val, "VAL")
test_lg, enc_rt  = logits_of(test, "TEST")
for it in val + test:                                # 600 Mo d'audio : plus besoin
    it.pop("array", None)
gc.collect()

def decode_all(lps, **kw):
    t0 = time.time()
    hyps = [beam_decode(lp, ID2TOK, BLANK, SPECIAL, delim=DELIM, **kw) for lp in lps]
    return hyps, (time.time() - t0) / len(lps)

# log_softmax du val calculé UNE fois : la grille rejoue ~24 décodages sur les
# mêmes clips, pas 24 conversions. (Le chiffre servi, lui, est mesuré sur le test
# avec la conversion incluse — c'est ce que le serveur paie vraiment.)
val_lp = [log_softmax(l.astype(np.float32).tolist()) for l in val_lg]

# ══ 3. référence : le greedy servi, sur CES logits, sur les 831 ══════════════
# v8 ne connaissait que l'estimation 0,6892 (torch complet + écart moteur mesuré
# sur 150 clips). Ici c'est le vrai chiffre du moteur servi.
g_val = [ctc_greedy(l.astype(np.float32).argmax(-1).tolist(), ID2TOK, BLANK, SPECIAL)
         for l in val_lg]
g_test = [ctc_greedy(l.astype(np.float32).argmax(-1).tolist(), ID2TOK, BLANK, SPECIAL)
          for l in test_lg]
gv = wer_cer([i["text"] for i in val], g_val)
gt = wer_cer([i["text"] for i in test], g_test)
print(f"GREEDY_VAL_WER={gv[0]:.4f} GREEDY_TEST_WER={gt[0]:.4f} "
      f"GREEDY_TEST_CER={gt[1]:.4f}", flush=True)
if abs(gt[0] - GREEDY_V8_EST) > 0.03:
    print(f"WARN_GREEDY_DRIFT: {gt[0]:.4f} vs estimation v8 {GREEDY_V8_EST:.4f} "
          "— la comparaison avec whisper reste valable (même test, même norm), "
          "mais l'estimation v8 était optimiste/pessimiste", flush=True)

# ══ 4. texte du LM : darija algérien public, JAMAIS les transcriptions de test ═
# Dialecte VÉRIFIÉ à la main (compteur de marqueurs راني/تاع/برك/وشراك contre
# ديال/غادي/كي-) : le titre d'un dataset ne prouve rien — Algerian-STT-Cleaned-V5
# annonce «Algerian» et sert du marocain (source unique = Adiren).
#   ayoubkirouane/Algerian-Darija = le plus dialectal trouvé (registre parlé) ;
#     ses 168 k lignes sont dans le SPLIT «v1», pas dans une config «v1».
#   DarijaDz = 3 M de lignes mais commentaires réseaux sociaux souvent MSA →
#     plafonné, sinon il noie les sources dialectales à 93 % du mélange.
# (nom, config, split, plafond en MOTS)
# Plafonds en MOTS, pas en lignes : une «ligne» du split train fait 646 mots en
# moyenne (ce sont des transcriptions entières, pas des phrases) — plafonner en
# lignes donnait 129 M de mots pour cette seule source, soit un dict de comptage
# de plusieurs centaines de Go et un LM servi de ~1,9 Go. Mesuré, pas supposé.
# À 12 M de mots : ~20 M de n-grammes avant élagage (~3 Go de pic) et ~3,5 M
# après min_count=2, soit ~42 Mo servis.
SOURCES = [("ayoubkirouane/Algerian-Darija", None, "train", 4_000_000),
           ("ayoubkirouane/Algerian-Darija", None, "v1", 3_000_000),
           ("touati-kamel/algerian-arabic-english-translation-50k", None, "train", 1_000_000),
           ("nasrellahkharroubi/DarijaDz", None, "train", 4_000_000)]

def text_column(d):
    cands = [c for c, f in d.features.items()
             if getattr(f, "dtype", None) == "string"]
    assert cands, "aucune colonne texte"
    pref = [c for c in cands if re.search(r"text|sentence|content|arab", c, re.I)]
    return (pref or cands)[0]

TEST_NORM = {i["text"] for i in test} | {i["text"] for i in val}
pool, dropped, parle = [], 0, []
for name, cfg, split, cap in SOURCES:
    try:
        d = load_dataset(name, cfg, split=split, token=HF_TOKEN)
        col = text_column(d)
        got = []
        for t in d[col]:
            t = norm(t or "")
            if t.count(" ") >= 1:                     # une phrase, pas un mot isolé
                if t in TEST_NORM:
                    dropped += 1                      # fuite du test : jetée et comptée
                else:
                    got.append(t)
        random.shuffle(got)
        pool += got[:cap]
        print(f"LM_SRC {name}/{split} col={col} {len(got)} → {min(len(got), cap)} lignes",
              flush=True)
        for s in got[:3]:
            print(f"    | {s[:110]}", flush=True)   # on lit ce qu'on nourrit au LM
    except Exception as e:
        print(f"WARN_LM_SRC_SKIPPED {name}/{split}: "
              f"{type(e).__name__}: {str(e)[:160]}", flush=True)

# ══ corpus PARLÉ v10 : le vrai corpus.txt (pas les 2724 lignes de metadata) ═══
# Le run précédent le chargeait via load_dataset → ne voyait que metadata.csv
# (2724 lignes) ; corpus.txt (55 573 lignes, registre parlé intégral) n'est pas
# une table, load_dataset ne le lit pas. On tire le zip directement.
import zipfile as _zf
from huggingface_hub import hf_hub_download as _dl
try:
    zp = _dl("Walidrbh27/khidmeti-stt-v10", "v10.zip", repo_type="dataset",
             token=HF_TOKEN)
    raw = _zf.ZipFile(zp).read("corpus.txt").decode("utf-8").splitlines()
    for t in raw:
        t = norm(t or "")
        if t.count(" ") >= 1:                     # une phrase, pas un mot isolé
            if t in TEST_NORM:
                dropped += 1                      # fuite test/val : jetée et comptée
            else:
                parle.append(t)
    print(f"LM_SRC khidmeti-stt-v10/corpus.txt PARLÉ {len(parle)} lignes "
          f"({sum(t.count(' ')+1 for t in parle)} mots)", flush=True)
    for s in parle[:3]:
        print(f"    | {s[:110]}", flush=True)
except Exception as e:
    print(f"WARN_LM_SRC_SKIPPED khidmeti-stt-v10/corpus.txt: "
          f"{type(e).__name__}: {str(e)[:160]}", flush=True)

# PLUS DE CONCATÉNATION. Deux corpus SÉPARÉS, pondérés au décodage (MIX_WEIGHTS).
# Deux raisons, les deux mesurées :
#   1. concaténer 55 k lignes à 12 M de mots donne au parlé 1,7 % des comptes →
#      le run précédent croyait tester le registre parlé, il ne testait rien
#      (test WER 0,6404 → 0,6419, du bruit).
#   2. BUG que ça supprime : `pool += parle` puis `random.shuffle(pool)` puis
#      `pool.pop()` jusqu'au plafond de mots pouvait effacer AU HASARD une partie
#      du corpus parlé — le contenu du LM dépendait du tirage.
assert len(pool) > 50_000, f"corpus écrit trop maigre : {len(pool)}"
random.shuffle(pool)
n_words = sum(t.count(" ") + 1 for t in pool)
while n_words > LM_WORDS:                             # filet RAM, cf. build_lm
    n_words -= pool.pop().count(" ") + 1

# ══ corpus DOMAINE : métiers × symptômes (lm_domaine.txt, S'ajoute au LM écrit) ═
# Petites phrases darija réelles des 3 tests terrain (الضوء مقطوع vs دور مقطوع,
# لافابو خاسرة, روبينيه تسيل…). AJOUTÉ APRÈS le plafond de mots et NON mélangé :
# le `while` ci-dessus (filet RAM) + le shuffle effaceraient au hasard ces lignes
# précieuses — exactement le bug parlé v9a. Le domaine dépasse le plafond de
# ~2 k mots sur 12 M : négligeable, et il est garanti dans le LM.
_DOM = """{{LM_DOMAINE}}""".splitlines()
_domaine = []
for t in _DOM:
    t = norm(t.strip() or "")
    if t.count(" ") >= 1 and t not in TEST_NORM:
        _domaine.append(t)
pool += _domaine
print(f"LM_SRC lm_domaine.txt DOMAINE {len(_domaine)} lignes "
      f"({sum(t.count(' ')+1 for t in _domaine)} mots, après plafond, non mélangé)",
      flush=True)
for s in _domaine[:3]:
    print(f"    | {s[:110]}", flush=True)

random.shuffle(parle)
n_words_p = sum(t.count(" ") + 1 for t in parle)
print(f"LM_CORPUS écrit lignes={len(pool)} mots={n_words}  "
      f"parlé lignes={len(parle)} mots={n_words_p}  "
      f"ratio={n_words / max(1, n_words_p):.0f}:1  fuites_test_jetées={dropped}",
      flush=True)

# ══ 5. grille (ordre, alpha, beta, plancher) sur le val ═══════════════════════
# Les logits sont en cache : un point de grille ne coûte qu'un décodage, donc on
# cherche large plutôt que de deviner. L'argmax reste la référence : à alpha=0,1
# et plancher désactivé on doit retomber près de lui.
print(f"=== grille sur val — {left()/3600:.1f} h ===", flush=True)
grid, best, lms, lms_p = [], None, {}, {}
for order in LM_ORDERS:
    t0 = time.time()
    lms[order] = build_lm(pool, order=order, min_count=LM_MINCOUNT)
    print(f"LM_BUILT order={order} ÉCRIT ngrams={len(lms[order]['keys'])} "
          f"en {(time.time()-t0)/60:.1f} min", flush=True)
    if parle:
        t0 = time.time()
        lms_p[order] = build_lm(parle, order=order, min_count=LM_MINCOUNT)
        print(f"LM_BUILT order={order} PARLÉ ngrams={len(lms_p[order]['keys'])} "
              f"en {(time.time()-t0)/60:.1f} min", flush=True)
    # w = poids du LM ÉCRIT. w=1.0 → écrit seul (le témoin, ce qui a produit le
    # record 0,6419) ; w<1 → le parlé pèse VRAIMENT, ce que la concaténation
    # n'avait jamais permis de tester.
    for w_mix in (MIX_WEIGHTS if parle else [1.0]):
        lmd = (lms[order] if w_mix >= 1.0 else
               {"order": order, "mix": (lms[order], lms_p[order], w_mix)})
        for a in ALPHAS:
            for b in BETAS:
                for fl in FLOORS:
                    hyps, ms = decode_all(val_lp, lm=lmd, alpha=a, beta=b,
                                          floor=fl, beam=BEAM0, prune=PRUNE)
                    wv = wer_cer([i["text"] for i in val], hyps)[0]
                    grid.append({"order": order, "w_mix": w_mix, "alpha": a,
                                 "beta": b, "floor": fl, "wer": round(wv, 4),
                                 "s_per_clip": round(ms, 3)})
                    if best is None or wv < best["wer"]:
                        best = grid[-1]
                        print(f"  NEW_BEST order={order} w_écrit={w_mix} a={a} "
                              f"b={b} floor={fl} val_WER={wv:.4f}", flush=True)
        print(f"  … order={order} w_écrit={w_mix} balayé "
              f"(meilleur courant {best['wer']:.4f}) — {left()/3600:.1f} h", flush=True)
    if left() < 2.5 * 3600:
        print("BUDGET: grille tronquée", flush=True)
        break
del pool, parle; gc.collect()

def lm_for(order, w_mix):
    """Le LM que désigne un point de grille — écrit seul ou mélange pondéré."""
    if w_mix >= 1.0:
        return lms[order]
    return {"order": order, "mix": (lms[order], lms_p[order], w_mix)}

LM_ORDER, W_MIX = best["order"], best["w_mix"]
lm = lm_for(LM_ORDER, W_MIX)
print(f"BEST_GRID order={LM_ORDER} w_écrit={W_MIX} alpha={best['alpha']} "
      f"beta={best['beta']} floor={best['floor']} val_WER={best['wer']:.4f} "
      f"(greedy val {gv[0]:.4f})", flush=True)

# Contrôle sur données RÉELLES de la récurrence : sans LM du tout, le faisceau
# n'est qu'un CTC, il doit égaler l'argmax (ou faire un cheveu mieux, il somme les
# alignements). S'il est nettement pire, le gain mesuré plus bas est du bruit sur
# un décodeur cassé, pas un gain. (Run v9a-1 : 0,6745 contre 0,6733 — bon.)
_w0 = wer_cer([i["text"] for i in val], decode_all(val_lp, lm=None)[0])[0]
print(f"BEAM_NO_LM_VAL_WER={_w0:.4f} (argmax {gv[0]:.4f})", flush=True)
if _w0 > gv[0] + 0.02:
    print(f"WARN_BEAM_WORSE_THAN_ARGMAX: {_w0:.4f} > {gv[0]:.4f} — récurrence "
          "suspecte, ne pas basculer sur ce chiffre", flush=True)

# largeur de faisceau : c'est le curseur précision/vitesse, on le mesure
sweep = []
for bw in BEAMS:
    hyps, ms = decode_all(val_lp, lm=lm, alpha=best["alpha"], beta=best["beta"],
                          floor=best["floor"], beam=bw, prune=PRUNE)
    w = wer_cer([i["text"] for i in val], hyps)[0]
    sweep.append({"beam": bw, "wer": round(w, 4), "s_per_clip": round(ms, 3)})
    print(f"  beam={bw} val_WER={w:.4f} ({ms:.2f} s/clip)", flush=True)
best_beam = min(sweep, key=lambda s: (round(s["wer"], 3), s["s_per_clip"]))["beam"]
print(f"BEST_BEAM={best_beam}", flush=True)
if best_beam == BEAMS[-1] and len(sweep) > 1 and sweep[-1]["wer"] < sweep[-2]["wer"]:
    print(f"WARN_BEAM_AT_GRID_EDGE: beam {best_beam} gagne ET progresse encore "
          f"({sweep[-2]['wer']:.4f} → {sweep[-1]['wer']:.4f}) — l'optimum est "
          "au-delà de la grille", flush=True)

# Ré-réglage alpha/beta AU FAISCEAU RETENU. La grille tourne à BEAM0=16 pour
# rester bon marché, mais on SERT à best_beam : à faisceau large un alpha plus
# fort devient sûr (le LM ne peut plus élaguer trop tôt la bonne hypothèse), donc
# l'optimum se déplace. Le run précédent réglait à 16 et servait à 32 sans jamais
# le remesurer = du gain laissé sur la table.
if best_beam != BEAM0:
    a0, b0 = best["alpha"], best["beta"]
    cand_a = sorted({a0, round(a0 * 1.5, 3), round(a0 * 2.0, 3), round(a0 * 3.0, 3)})
    cand_b = sorted({0.0, b0, b0 + 1.0})
    print(f"=== ré-réglage à beam={best_beam} : alpha={cand_a} beta={cand_b} "
          f"— {left()/3600:.1f} h ===", flush=True)
    for a in cand_a:
        for b in cand_b:
            hyps, ms = decode_all(val_lp, lm=lm, alpha=a, beta=b,
                                  floor=best["floor"], beam=best_beam, prune=PRUNE)
            wv = wer_cer([i["text"] for i in val], hyps)[0]
            grid.append({"order": LM_ORDER, "w_mix": W_MIX, "alpha": a, "beta": b,
                         "floor": best["floor"], "beam": best_beam,
                         "wer": round(wv, 4), "s_per_clip": round(ms, 3)})
            if wv < best["wer"]:
                best = grid[-1]
                print(f"  NEW_BEST_AT_BEAM a={a} b={b} val_WER={wv:.4f}", flush=True)
    print(f"BEST_TUNED alpha={best['alpha']} beta={best['beta']} "
          f"val_WER={best['wer']:.4f} @beam={best_beam}", flush=True)

# ══ 6. gate : le test complet, moteur servi, décodage retenu ══════════════════
print(f"=== test 831 clips — {left()/3600:.1f} h ===", flush=True)
# Conversion incluse dans le chrono : c'est exactement le travail du serveur.
t0, hyps = time.time(), []
for k, l in enumerate(test_lg):
    hyps.append(beam_decode(log_softmax(l.astype(np.float32).tolist()),
                            ID2TOK, BLANK, SPECIAL, lm=lm, alpha=best["alpha"],
                            beta=best["beta"], floor=best["floor"],
                            beam=best_beam, prune=PRUNE, delim=DELIM))
    if k and k % 200 == 0:
        print(f"  [BEAM] {k}/{len(test_lg)} — {left()/3600:.1f} h restantes", flush=True)
dec_s = (time.time() - t0) / len(test_lg)
bw, bc = wer_cer([i["text"] for i in test], hyps)
dur_moy = sum(i["dur"] for i in test) / len(test)
tot_rt = enc_rt + dec_s / dur_moy
gate_pass = bw < SERVED_WER
print(f"[BEAM] hyp0: {hyps[0][:120]}", flush=True)

bins = {}
for lo, hi in [(0.5, 5), (5, 10), (10, 20), (20, 30)]:
    idx = [i for i, it in enumerate(test) if lo < it["dur"] <= hi]
    if idx:
        w2, c2 = wer_cer([test[i]["text"] for i in idx], [hyps[i] for i in idx])
        bins[f"{lo}-{hi}s"] = {"n": len(idx), "wer": round(w2, 4), "cer": round(c2, 4)}

print("\n┌─ récap (mêmes 831 clips, même norm, même moteur int8 4 threads) ─")
print(f"│ whisper medium+LoRA SERVI ........... WER {SERVED_WER:.4f}  6,044×RT")
print(f"│ CTC v8 greedy (argmax) ............. WER {gt[0]:.4f} CER {gt[1]:.4f} "
      f"{enc_rt:.3f}×RT")
print(f"│ CTC v8 + LM {LM_ORDER}-gram faisceau {best_beam} .... WER {bw:.4f} CER {bc:.4f} "
      f"{tot_rt:.3f}×RT")
print(f"│ gain du LM ......................... {gt[0]-bw:+.4f} WER "
      f"({(gt[0]-bw)/max(1e-9,gt[0])*100:+.1f} % relatif)")
print(f"│ coût du décodage ................... {dec_s*1000:.0f} ms/clip "
      f"(clip moyen {dur_moy:.1f} s)")
print(f"│ vitesse vs whisper servi ........... ×{6.044/max(1e-9,tot_rt):.1f}")
print(f"│ WER par durée ...................... {bins}")
print("└─")
print(f"GATE_BEATS_SERVED={'YES' if gate_pass else 'NO'} ({bw:.4f} vs {SERVED_WER:.4f})")
print(f"GATE_PASS={'YES' if gate_pass else 'NO'}", flush=True)

# ══ 6b. diagnostic : OÙ vit le WER restant ════════════════════════════════════
# Le CER du CTC+LM bat déjà whisper (0,2361 contre 0,2373) alors que le WER perd
# 1,2 pt : les CARACTÈRES sont bons, les MOTS sont faux. Deux causes possibles, et
# il faut savoir LAQUELLE avant de dépenser un run de plus :
#   (a) désaccord de SEGMENTATION/orthographe — la darija n'a ni orthographe ni
#       découpage standard, l'annotateur écrit «ما راهش» ou «ماراهش» au choix.
#       Une soudure coûte DEUX erreurs de mots et zéro erreur de caractère : c'est
#       exactement la signature observée. → un post-traitement de segmentation.
#   (b) mots réellement mal reconnus → seule la donnée ou un meilleur LM aide.
# Et l'OOV : si le LM ignore une grosse part du vocabulaire de RÉFÉRENCE, il ne
# PEUT pas aider — ce qui expliquerait à la fois l'alpha optimal si bas (0,2) et
# le gain plafonné à 5 % relatif là où la littérature annonce 15-25 %.
def unspace(s):
    return s.replace(" ", "")

def lm_has(lmd, word):
    """Le mot est-il un unigramme connu du LM ? (mélange : connu de l'un des deux)"""
    mx = lmd.get("mix")
    if mx:
        return lm_has(mx[0], word) or lm_has(mx[1], word)
    k, keys = _h(word), lmd["keys"]
    j = bisect.bisect_left(keys, k)
    return j < len(keys) and keys[j] == k

R = [norm(i["text"]) or "-" for i in test]
H = [norm(h) or "-" for h in hyps]
n_ref_w = sum(len(r.split()) for r in R)
n_hyp_w = sum(len(h.split()) for h in H)
err_w = [jiwer.wer(r, h) * max(1, len(r.split())) for r, h in zip(R, H)]
tot_err = sum(err_w)
exact = sum(1 for r, h in zip(R, H) if r == h)
# Énoncés dont les caractères concordent EXACTEMENT mais pas le découpage :
# 100 % de leurs erreurs de mots sont de la segmentation, sans hypothèse.
seg_idx = [i for i in range(len(R)) if unspace(R[i]) == unspace(H[i]) and R[i] != H[i]]
seg_err = sum(err_w[i] for i in seg_idx)
cer_sp = jiwer.cer(R, H)
cer_nosp = jiwer.cer([unspace(r) for r in R], [unspace(h) or "-" for h in H])
vocab_ref = {w for r in R for w in r.split()}
oov_ty = sum(1 for w in vocab_ref if not lm_has(lm, w))
oov_tk = sum(1 for r in R for w in r.split() if not lm_has(lm, w))

print("\n┌─ diagnostic : où vit le WER restant ─")
print(f"│ énoncés exacts ..................... {exact}/{len(R)} "
      f"({exact / len(R) * 100:.1f} %)")
print(f"│ énoncés à SEGMENTATION seule ....... {len(seg_idx)}/{len(R)} "
      f"(mêmes caractères, découpage différent)")
print(f"│ part du WER due à la segmentation .. {seg_err / max(1e-9, tot_err) * 100:.1f} % "
      f"({seg_err:.0f} / {tot_err:.0f} erreurs de mots)")
print(f"│ CER avec espaces ................... {cer_sp:.4f}")
print(f"│ CER SANS espaces (borne basse) ..... {cer_nosp:.4f} → les espaces "
      f"coûtent {cer_sp - cer_nosp:+.4f}")
print(f"│ mots hyp / mots réf ................ {n_hyp_w}/{n_ref_w} = "
      f"{n_hyp_w / max(1, n_ref_w):.3f} "
      f"({'tendance SOUDURE' if n_hyp_w < n_ref_w else 'tendance DÉCOUPAGE'})")
print(f"│ OOV du LM sur la référence ......... types {oov_ty}/{len(vocab_ref)} "
      f"({oov_ty / max(1, len(vocab_ref)) * 100:.1f} %) · tokens {oov_tk}/{n_ref_w} "
      f"({oov_tk / max(1, n_ref_w) * 100:.1f} %)")
print("└─", flush=True)

# Hypothèses sauvées : plus jamais besoin d'une passe encodeur (15 min CPU) pour
# étudier la structure d'erreur — quatre runs l'ont déjà payée.
os.makedirs("out", exist_ok=True)
json.dump({"refs": R, "hyps": H,
           "config": {"order": LM_ORDER, "w_ecrit": W_MIX, "alpha": best["alpha"],
                      "beta": best["beta"], "floor": best["floor"],
                      "beam": best_beam},
           "wer": round(bw, 4), "cer": round(bc, 4),
           "diag": {"exact": exact, "seg_only_utts": len(seg_idx),
                    "seg_share_of_wer": round(seg_err / max(1e-9, tot_err), 4),
                    "cer_spaced": round(cer_sp, 4), "cer_unspaced": round(cer_nosp, 4),
                    "hyp_ref_word_ratio": round(n_hyp_w / max(1, n_ref_w), 4),
                    "lm_oov_types": round(oov_ty / max(1, len(vocab_ref)), 4),
                    "lm_oov_tokens": round(oov_tk / max(1, n_ref_w), 4)}},
          open("out/hyps.json", "w"), ensure_ascii=False)

# ══ 7. artefacts ══════════════════════════════════════════════════════════════
# Le LM retenu peut être un MÉLANGE (écrit × parlé) : il n'a alors pas de "keys"
# à lui, il pointe deux LM. On sauve donc les composants + le poids, et on
# reconstruit le mélange pour l'aller-retour — sinon np.savez planterait ici,
# après le gate, en perdant tout le run.
os.makedirs("out", exist_ok=True)

def save_lm(lmd, path):
    np.savez_compressed(path,
                       keys=np.asarray(lmd["keys"], dtype=np.int64),
                       logp=np.asarray(lmd["logp"], dtype=np.float32),
                       order=lmd["order"], backoff=lmd["backoff"],
                       oov=lmd["oov"], n_tok=lmd["n_tok"])

IS_MIX = bool(lm.get("mix"))
if IS_MIX:
    save_lm(lm["mix"][0], "out/lm.npz")            # écrit
    save_lm(lm["mix"][1], "out/lm_parle.npz")      # parlé
    _rt = {"order": lm["order"],
           "mix": (dict(np.load("out/lm.npz")), dict(np.load("out/lm_parle.npz")),
                   lm["mix"][2])}
else:
    save_lm(lm, "out/lm.npz")
    _rt = dict(np.load("out/lm.npz"))
# Parité de l'ARTEFACT : le serveur charge ces .npz, pas le dict Python. Un int64
# mal converti ou un tri perdu ferait renvoyer à bisect la mauvaise entrée, en
# silence, avec un WER dégradé qu'on mettrait sur le dos du LM.
_probe = []
for it in test[:60]:
    w = it["text"].split()
    if len(w) >= 3:
        _probe.append((tuple(w[:2]), w[2]))
_probe = _probe[:10] + [(("زززز",), "زززز")]
for _hh, _ww in _probe:
    a, b = lm_logp(lm, _hh, _ww), lm_logp(_rt, _hh, _ww)
    assert abs(a - b) < 1e-4, (_hh, _ww, a, b)
print(f"LM_ROUNDTRIP_OK ({len(_probe)} sondes, mélange={IS_MIX})", flush=True)
meta = dict(meta_v8)
meta.update({
    "version": "v10-ctc-lm",
    "decode": "faisceau de préfixes CTC + fusion superficielle n-gram stupid-backoff ; "
              "id2tok/blank_id/delim viennent de ctc_vocab.json, lm.npz porte "
              "(keys int64 triés 63 bits, logp float32) — voir beam_decode() ; "
              "floor = plancher du score LM par mot ; si w_ecrit < 1, le score est "
              "l'interpolation LOG-LINÉAIRE w·lm.npz + (1−w)·lm_parle.npz",
    "lm": {"order": LM_ORDER, "w_ecrit": W_MIX, "is_mix": IS_MIX,
           "min_count": LM_MINCOUNT,
           "words_ecrit": n_words, "words_parle": n_words_p,
           "ngrams": int(len(lm["mix"][0]["keys"] if IS_MIX else lm["keys"])),
           "sources": [f"{n}/{s}≤{c}" for n, _, s, c in SOURCES]
                     + ["khidmeti-stt-v10/corpus.txt (parlé, LM séparé)"],
           "test_lines_dropped": dropped,
           "alpha": best["alpha"], "beta": best["beta"], "floor": best["floor"],
           "beam": best_beam, "prune": PRUNE},
    "wer_norm": {**meta_v8.get("wer_norm", {}),
                 "served_whisper_v3": SERVED_WER,
                 "test_int8_greedy": round(gt[0], 4),
                 "test_int8_beam_lm": round(bw, 4),
                 "val_greedy": round(gv[0], 4), "val_beam_lm": round(best["wer"], 4),
                 "val_beam_no_lm": round(_w0, 4)},
    "cer_norm": {**meta_v8.get("cer_norm", {}),
                 "test_int8_greedy": round(gt[1], 4),
                 "test_int8_beam_lm": round(bc, 4)},
    "wer_by_duration": bins,
    "xrt_cpu_4threads": {**meta_v8.get("xrt_cpu_4threads", {}),
                         "encoder": round(enc_rt, 4),
                         "decode_s_per_clip": round(dec_s, 3),
                         "total_beam": round(tot_rt, 4)},
    "grid": grid, "beam_sweep": sweep,
    "gate": f"WER(faisceau+LM, int8, 831 clips) < {SERVED_WER}",
    "serve_file": SERVE_FILE,
})
json.dump(meta, open("out/meta.json", "w"), indent=2, ensure_ascii=False)

api = HfApi(token=HF_TOKEN)
if not gate_pass:
    api.upload_folder(folder_path="out", repo_id=CAND_REPO)   # trace, prod intouchée
    print(f"GATE_FAILED_NO_SWITCH — whisper reste servi ; LM + mesures sur {CAND_REPO}")
    print(f"V9A: greedy {gt[0]:.4f} → LM {bw:.4f} (servi {SERVED_WER:.4f}), "
          f"{tot_rt:.3f}×RT en {(time.time()-T0)/3600:.1f} h")
    sys.exit(0)

import shutil
shutil.copy(pull(SERVE_FILE), f"out/{SERVE_FILE}")
shutil.copy(pull("ctc_vocab.json"), "out/ctc_vocab.json")
api.create_repo(REPO, private=True, exist_ok=True)
api.upload_folder(folder_path="out", repo_id=REPO)
print("HF_UPLOAD_DONE=" + REPO)
# Auto-restart Space to pick up new weights
try:
    from huggingface_hub import HfApi as _HfApi2
    _HfApi2(token=HF_TOKEN).restart_space(repo_id="Walidrbh27/khidmeti-ai", repo_type="space")
    print("SPACE_RESTARTED=Walidrbh27/khidmeti-ai")
except Exception as _e:
    print(f"SPACE_RESTART_SKIP={_e}")

print("\n── à mettre dans .env.cloud ET .env.local (jamais le .env généré) ──")
print("STT_ENGINE=ctc")
print(f"STT_MODEL={REPO}")
print(f"STT_MODEL_FILE={SERVE_FILE}")
print(f"STT_LM_ALPHA={best['alpha']}  STT_LM_BETA={best['beta']}  "
      f"STT_LM_FLOOR={best['floor']}  STT_LM_BEAM={best_beam}")
print("# retour arrière : STT_ENGINE=whisper (le repo whisper reste intact)")
print(f"V9A COMPLETE: {gt[0]:.4f} → {bw:.4f} (servi {SERVED_WER:.4f}), "
      f"{tot_rt:.3f}×RT en {(time.time()-T0)/3600:.1f} h")
