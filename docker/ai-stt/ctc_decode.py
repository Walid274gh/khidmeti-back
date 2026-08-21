# docker/ai-stt/ctc_decode.py
#
# Décodeur CTC à faisceau de préfixes + fusion superficielle n-gram (stupid
# backoff, Brants et al. 2007). Copie IDENTIQUE de la source de vérité
# ml/kaggle/stt_lm_kernel_v9.py (sections _h / build_lm / lm_logp / log_softmax /
# beam_decode + demo()) : le WER servi (0,6236 sur les 831 clips du test, modèle
# 24 h B_ep8, LM ordre 3 écrit) a été MESURÉ avec ce décodage — toute divergence
# ici casse silencieusement ce chiffre.
#
# Stdlib uniquement pour décoder (math, bisect, hashlib, struct, zipfile) ;
# numpy est utilisé SANS obligation pour charger les .npz du LM (le serveur l'a
# de toute façon via onnxruntime). load_lm() retombe sur un lecteur .npy maison
# si numpy manque — même contenu, même ordre des clés.
#
# Paramètres de prod (mesurés au run khidmeti-stt-lm-v10, BEST_TUNED) :
#   order=3 (w_écrit=1.0 → LM écrit seul, pas de mélange)  alpha=0.3  beta=2.0
#   floor=-30.0  beam=128  prune=1e-4  TOPK=8

import bisect
import hashlib
import math
import struct
import zipfile

NEG = -1e30
TOPK = 8          # plafond de sécurité : avec prune=1e-4 il y a 1 à 3 candidats
DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_FLOOR, DEFAULT_BEAM = 0.3, 2.0, -30.0, 128


def _h(s):
    # 63 bits, pas 64 : les clés doivent tenir dans un int64 signé, sinon un
    # tableau numpy uint64 comparé à un int Python peut passer par float64 et
    # bisect renvoie silencieusement la mauvaise entrée.
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(),
                          "little") & 0x7FFFFFFFFFFFFFFF


def build_lm(lines, order=3, min_count=2):
    """n-gram à backoff bête, stocké en (hachages triés, log-scores) : bisect
    marche sur une liste comme sur un tableau numpy, donc le serveur charge un
    .npz sans changer une ligne d'ici."""
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


def log_softmax(rows):
    """Logits bruts (liste de listes) → log-probabilités."""
    out = []
    for r in rows:
        m = max(r)
        s = math.log(sum(math.exp(x - m) for x in r))
        out.append([x - m - s for x in r])
    return out


def beam_decode(rows, id2tok, blank, special, lm=None, alpha=0.0, beta=0.0,
                beam=16, prune=1e-4, delim="|", floor=-30.0):
    """rows = sortie de log_softmax. État = (mots finis, mot en cours) : c'est
    exactement le texte rendu, donc deux alignements du même texte fusionnent.
    floor = plancher du score LM par mot. Sans lui, un mot juste mais absent du
    corpus coûte log(1/N) ≈ −16 nats, et COLLER deux mots ne paie qu'un seul
    malus au lieu de deux : le LM soude les mots (pire faute possible en WER)."""
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


def ctc_greedy(ids, id2tok, blank, special):
    out, prev = [], -1
    for i in ids:
        if i != prev and i != blank:
            t = id2tok[i]
            if t not in special:
                out.append(t)
        prev = i
    return " ".join("".join(out).replace("|", " ").split())


# ══ chargement des .npz du LM ═══════════════════════════════════════════════
# save_lm() (kernel) écrit : keys int64 triés, logp float32, puis scalaires
# order/backoff/oov/n_tok. numpy : chemin normal du serveur. Repli stdlib :
# un .npz est un zip de .npy ; on lit l'en-tête puis les octets bruts
# little-endian.

def _load_npz_pure(path):
    """No numpy : lit keys/logp + scalaires du .npz de save_lm(). Les .npy de
    np.savez_compressed sont en version 1 tant que le header tient en 64 ko
    (notre cas : dictionnaire dtype+shape court) → longueur sur 2 octets."""
    import ast
    out = {}
    with zipfile.ZipFile(path) as zf:
        for name in ["keys.npy", "logp.npy", "order.npy",
                     "backoff.npy", "oov.npy", "n_tok.npy"]:
            with zf.open(name) as f:
                assert f.read(6) == b"\x93NUMPY"
                major, minor = f.read(1)[0], f.read(1)[0]
                hlen = struct.unpack("<H" if major == 1 else "<I", f.read(2 if major == 1 else 4))[0]
                hdr = f.read(hlen)
                d = ast.literal_eval(hdr[:hdr.index(b"}") + 1].decode("ascii"))
                descr, shape = d["descr"], d["shape"]
                n = 1
                for s in shape:
                    n *= s
                if name == "keys.npy":
                    assert descr.endswith("i8"), descr
                    raw = f.read(8 * n)
                    out["keys"] = [struct.unpack("<q", raw[i * 8:(i + 1) * 8])[0]
                                   for i in range(n)]
                elif name == "logp.npy":
                    assert descr.endswith(("f4", "f8")), descr
                    wid = 4 if descr.endswith("f4") else 8
                    raw = f.read(wid * n)
                    out["logp"] = [struct.unpack("<f" if wid == 4 else "<d",
                                                 raw[i * wid:(i + 1) * wid])[0]
                                   for i in range(n)]
                elif name == "order.npy":
                    out["order"] = int(struct.unpack("<q", f.read(8))[0])
                elif name == "backoff.npy":
                    out["backoff"] = struct.unpack("<d", f.read(8))[0]
                elif name == "oov.npy":
                    out["oov"] = struct.unpack("<d", f.read(8))[0]
                elif name == "n_tok.npy":
                    out["n_tok"] = int(struct.unpack("<q", f.read(8))[0])
    assert out["keys"] == sorted(out["keys"]), "clés du LM non triées"
    assert len(out["keys"]) == len(out["logp"])
    return out


def load_lm(path):
    try:
        import numpy as np
        d = dict(np.load(path, allow_pickle=False))
        return {"keys": np.asarray(d["keys"], dtype=np.int64),
                "logp": np.asarray(d["logp"], dtype=np.float32),
                "order": int(d["order"]), "backoff": float(d["backoff"]),
                "oov": float(d["oov"]), "n_tok": int(d["n_tok"])}
    except ImportError:
        return _load_npz_pure(path)


def load_lm_cfg(path_ecrit, path_parle, meta_lm):
    """Construit le dict LM à SERVIR depuis les .npz + la config meta.json.
    meta_lm = meta.json["lm"] (écrit par le kernel) : order, is_mix, w_ecrit…
    Le serveur charge CE QUE LE KERNEL A MESURÉ — jamais un ordre en dur."""
    order = int(meta_lm.get("order", 3))
    if meta_lm.get("is_mix") and path_parle:
        a, b = load_lm(path_ecrit), load_lm(path_parle)
        w = float(meta_lm.get("w_ecrit", 0.5))
        lm = {"order": order, "mix": (a, b, w)}
        _lms = (a, b)
    else:
        lm = load_lm(path_ecrit)
        if lm["order"] != order:           # tolérant : l'ordre vient du .npz
            order = lm["order"]
        _lms = (lm,)
    for _l in _lms:
        assert _l["backoff"] == math.log(0.4), "LM non stupid-backoff"
    return lm


# ══ démo de parité (reprise VERBATIM du kernel) ═════════════════════════════
def demo():
    import random
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

    # 1. sur des logits piqués, le faisceau DOIT retomber sur l'argmax.
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
    assert lm["keys"] == sorted(lm["keys"])

    # 5. alpha/beta à zéro = le LM n'influence RIEN (le no-op le plus testé qui soit)
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
    #    seul, entre les deux ça interpole VRAIMENT — deux LM qui se
    #    CONTREDISENT doivent renverser le décodage selon w.
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
    print(f"ctc_decode.demo OK (7 tests, LM {lm['order']}-gram, {len(lm['keys'])} ngrams)")


if __name__ == "__main__":
    demo()