# STT CTC — pipeline complet, état FINAL servit (Aug 20 2026)

Réentraînement du wav2vec2-XLSR darija (CTC) sur les 56k clips algériens
vérifiés. La chasse au WER est terminée et MESURÉE de bout en bout : le CTC
**est désormais servi en production** (décision produit du 20/08 — latence),
même si le gate WER dit non. Ce fichier est la référence pour relancer le
pipeline (ex. l'extension 93 h) avec les bons chiffres.

## Verdicts finaux (tous mesurés sur les 831 clips Casablanca-Algeria, même norm)

| moteur | WER | CER | latence |
|---|---|---|---|
| whisper medium+LoRA (réglages PROD : vad+prompt, int8 4 threads) | **0.6097** IC[0.5930;0.6276] | 0.2368 | 6.044×RT (60 s / clip 10 s) |
| CTC 24h greedy (B_ep8, int8) | 0.6688 | 0.2405 | 0.169×RT |
| **CTC 24h + LM 3-gram (servi)** | **0.6236** | **0.2310** | **0.231×RT (×26 vs whisper)** |

- Bootstrap apparié (4000 tirages, σ=0.0080) : écart CTC(6h+LM) − whisper =
  +0.0292 réel, IC [+0.0130;+0.0447], zéro exclu. La constante du gate a été
  mesurée fausse (0.6300 = estimation sans vad+prompt) et **corrigée à 0.6097**
  en dur dans `stt_lm_kernel_v9.py` (ligne 49) et documentée ici.
- Leviers de décodage TOUS morts par mesure : beam (128 optimal au modèle 24h,
  retourne différemment selon le modèle — ne jamais extrapoler), alpha/beta
  ré-réglés à beam 128 (0.3/2.0), floor à zéro effet (désactivé, −30), OOV LM
  5.4 % tokens, segmentation 0.4 % du WER, 17.4 pts de WER = variantes de
  graphie à 1 caractère (irréductibles, valent pour whisper aussi).

## Sources 56k vérifiées (marqueurs dialecte, session v9a — noms complets)

| HF dataset | config | clips | texte | durée |
|---|---|---|---|---|
| `oddadmix/arabic-audio-collection-algerian-kahwa-postcast` | default | 23 264 | `transcript_text` | colonne |
| `yasminekaced/algerian_tts` | default | 23 356 | `text` | colonne |
| `oddadmix/arabic-audio-collection-algerian-rawi` | default | 5 296 | `transcript_text` | colonne |
| `FatimahEmadEldin/cafe-algerian-codeswitch-speech` | large | 3 830 | `transcription` | décodage (pas de colonne) |

~30 Go de parquet (audio embarqué) — ne tiennent pas sur le disque d'un kernel
(≈19,5 Go) → sélection une fois, en CPU, dans un dataset Kaggle privé.
Version **93 h** (Aug 20 2026) : `STAGE_B_HOURS=90.5`,
`SOURCE_TARGET_HOURS={kahwa:50, dztv:33, rawi:6, cafe:1.5}` (dispos mesurées
50.6/34.0/6.4/1.9), assert plage **80–95 h**, ~40 k clips FLAC ≈ **6 Go**
(v10.zip, STORED) — flac ÷3 le transfert (EOF TLS Kaggle vu avec du wav).
Historique : 24 h = 10 900 clips / 1,59 Go ; 6 h = 2728 clips / 230 Mo.

## Pipeline (2 kernels + 1 dataset)

1. **Prep (CPU, quota GPU préservé)** — `stt_prep_v10.py` (v15) :
   dztv chargée EN PREMIER (fragile), dédoublonnage, garde-fou anti-fuite
   (test 831), parts sources **24 h** (kahwa 10 · dztv 10 · rawi 3 · cafe 1,
   plafonds en SECONDES, asserts 20 ≤ total ≤ 26 h), `corpus.txt` = tout le
   texte scanné (55 573 lignes, registre parlé pour le LM), upload dataset HF
   privé `Walidrbh27/khidmeti-stt-v10`.
   ```bash
   CPU=1 python3 stt_push.py push stt_prep_v10.py khidmeti-stt-prep-v10
   ```
2. **Train (GPU T4)** — `stt_kernel_v10.py` : v8 verbatim (SEED 20260817,
   val/test 83/831 bit-identiques), étage A 1 ép, étage B = DZ ×2 + v10
   (assert 80–95 h). Le dataset v10 arrive par snapshot_download + unzip +
   **rmtree des copies hf_v10/hf_cache_v10** (6 Go × 3 ne tient pas sur
   19,5 Go). ⚠️ Budget 7.6 h fixe : ~3 h/époque à 90 h → **~2 époques de B**
   (échange assumé : 90 h uniques vues 2× > 24 h vues 8×). Résultats mesurés
   24 h (pour référence) : BEST B_ep8 val 0.6580, greedy test 0.6625 (torch)
   / 0.6688 (int8+écart moteur), +LM 0.6236.
   ```bash
   python3 stt_push.py push stt_kernel_v10.py khidmeti-stt-retrain-v10
   ```
2b. **Séquence à respecter** : push du train et du LM SEULEMENT après la fin
   du prep — le prep RÉÉCRIT v10.zip sur HF (6 Go) ; lire le zip en parallèle
   (= LM kernel, qui tire corpus.txt du zip) risque un cache HF corrompu.
3. **LM + gate** — `stt_lm_kernel_v9.py` v4 (CPU, constante 0.6097) : recharge
   les poids de `khidmeti-stt-ctc-cand`, grille order 3/4 + alpha/beta + beam,
   ré-réglage à beam choisi, écrit `out/` sur le repo candidat. Résultat 24h :
   order 3 écrit seul (w_écrit=1.0), alpha 0.3, beta 2.0, beam 128 → **0.6236**.
   ```bash
   CPU=1 python3 stt_push.py push stt_lm_kernel_v9.py khidmeti-stt-lm-v10
   ```
4. **Bascule — APPLIQUÉE le 2026-08-20** (décision produit) — dans
   `.env.cloud` ET `.env.local` (jamais le `.env` généré) :
   ```
   STT_ENGINE=ctc
   STT_MODEL=Walidrbh27/khidmeti-stt-ctc      # repo STABLE (restauré 63c09cef)
   STT_MODEL_FILE=model.int8.onnx
   ```
   ⚠️ **CONFIG LM = meta.json du repo, jamais en dur** : le serveur lit
   order/mix/α/β/floor/beam de `meta.json["lm"]` à chaque boot (env STT_LM_* =
   override ponctuel seulement, compose les passe VIDE par défaut). Bug payé le
   20/08 : le run 93h a écrit un LM ordre-4 mélangé sur `-cand`, l'ancien
   serveur "assert order==3" plantait → conteneur en boucle → FALLBACK → 0% en
   prod. Le repo SERVÉ est `khidmeti-stt-ctc` (stable, restauré depuis l'état
   connu-bon `63c09cef1612` via `stt_restore_served.py`) — `-cand` reste le
   bac à sable des runs, jamais pointé par la prod.
   Le serveur (`docker/ai-stt/server.py` + `ctc_decode.py` — portage fidèle du
   décodeur kernel, démo 7 tests au boot) fait le faisceau+LM au lieu du
   greedy. Retour arrière = commenter les lignes `STT_ENGINE` et remettre
   `STT_MODEL=Walidrbh27/khidmeti-stt` : le repo whisper reste intact.

## Suivi

```bash
./poll_kernels.sh khidmeti-stt-prep-v10 khidmeti-stt-retrain-v10 khidmeti-stt-lm-v10
python3 stt_push.py log khidmeti-stt-lm-v10 | tr '\r' '\n' | grep -E "GATE|V9A|BEST|WER"
```

## Garde-fous câblés

- Comparabilité : SEED et ORDRE des tirages identiques à v8 → val/test
  bit-identiques.
- `assert 20 ≤ heures v10 ≤ 26` dans le train : un prep partiel échoue avant
  l'entraînement.
- self-check prep : comptes flac == lignes metadata, un flac re-lu mesure sa
  durée.
- Licences : oddadmix « other », Casablanca CC BY-NC-ND → poids intérimaires
  non-commerciaux, remplacer par flywheel avant commercialisation.
- Transport data v10 = HF (le multipart Kaggle coupe en EOF TLS).
- Côté serveur : `assert lm["order"]==3` + backoff=log(0.4) au boot ; le LM
  chargeable en stdlib pur si numpy manquait (mêmes valeurs).