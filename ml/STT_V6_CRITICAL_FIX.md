# Khidmeti AI Training v6 — CRITICAL FIX + Massive Data Expansion

**Date:** 2026-08-10
**Status:** ⛔ **DÉPASSÉ / متجاوز — le volet STT de ce document est FAUX. Ne pas s'en servir.**

> ## ⛔ Correction du 15 août 2026 — mesures, pas espoirs
>
> Le plan STT v6 ci-dessous n'a **jamais tourné** et sa prémisse est fausse. Ce
> qui a été **mesuré** depuis (run v7, test set Casablanca-Algeria 831 clips,
> moteur de prod faster-whisper int8, normalisation identique à v3) :
>
> | mesure | WER | CER |
> |---|---|---|
> | v3 servi (whisper-medium + LoRA) | **0,6300** | 0,2373 |
> | `openai/whisper-large-v3` zero-shot | **0,8055** | 0,3708 |
> | large-v3 + LoRA — meilleur val (epoch 3/5) | **0,5738** | 0,2439 |
>
> Trois erreurs à ne pas répéter :
>
> 1. **`anaszil/whisper-large-v3-turbo-darija` n'est pas de la darija
>    algérienne** et n'est pas non plus un modèle servable : c'est un **adaptateur
>    PEFT/LoRA**, que faster-whisper ne sait pas charger. Mis dans `STT_MODEL`,
>    il mettait le conteneur `ai-stt` en FALLBACK silencieux (corrigé).
> 2. **« <0,40 WER attendu » n'a jamais été une mesure**, seulement un espoir.
>    L'écrire comme un fait est exactement ce qu'on ne fait pas.
> 3. **Changer de socle ne suffit pas** : large-v3 nu est *pire* (0,8055) que
>    notre medium fine-tuné (0,6300). Le gain vient du fine-tuning sur données
>    algériennes, pas de la taille du socle.
>
> Référence à jour : `ml/kaggle/stt_kernel_v7b.py`.

---

## 🔥 CRITICAL DISCOVERY: STT Was Using Wrong Base Model!

**المشكلة السابقة:**
- كنا نبدأ من `openai/whisper-medium` الخام
- هذا النموذج **غير مدرب على الدارجة** (فقط فصحى + لغات أخرى)
- لهذا كان يسمع "blombai" بدلاً من "plombier/بلومبيي"

**الحل v6:** ⛔ *(خاطئ — انظر التصحيح أعلى الملف)*
استخدام **anaszil/whisper-large-v3-turbo-darija** كنموذج أساسي
- ❌ **ليس مدرباً على الدارجة الجزائرية** — والأهم: إنه adaptateur PEFT، لا يمكن لـ faster-whisper تحميله
- ✅ whisper v3 turbo (أحدث + أسرع من v1/v2)
- ✅ 1,182 تحميل على HuggingFace (مثبت)
- ✅ مارس 2025 (حديث جداً)
- ✅ MIT license (حر تماماً)

**التأثير المتوقع:** ⛔ *(لم يُقَس أبداً — رقم مأمول لا مقيس)*
- **v3 baseline:** 0.6303 WER (من whisper-medium خام)
- **v6 expected:** <0.40 WER — **هذا لم يُقَس، والمقيس فعلاً 0.8055 للسوكل النقي**

---

## NLU v6: Data Expansion (Unchanged)

**70,408 rows** (10.7× من 6,581 baseline)
- Stratified splits: 59,860 train / 3,513 val / 7,035 test
- Augmentation: regional dialects + arabizi + contexts + typos
- **Gate: prof_acc ≥ 0.90**

---

## STT v6: Fixed Base Model + Enhanced Training

### Base Model Change (Most Important!) — ⛔ faux, voir en-tête
```python
# OLD (WRONG):
MEDIUM = "openai/whisper-medium"  # NOT trained on Darija!

# ⛔ v6 (WRONG TOO — adaptateur PEFT, faster-whisper ne le charge pas) :
BASE_MODEL = "anaszil/whisper-large-v3-turbo-darija"

# ✅ ce qui est réellement utilisé (stt_kernel_v7b.py) :
BASE_MODEL = "openai/whisper-large-v3"   # nu 0,8055 ; +LoRA algérien 0,5738 val
```

### Training Adjustments for Larger Model
- **Base:** whisper-medium (769M params) → v3-turbo-large (809M params, optimized)
- **LoRA rank:** 32 → 64 (more capacity)
- **Batch size:** 4 → 2 (larger model needs less batch)
- **Grad accum:** 8 → 16 (compensate smaller batch)
- **Learning rate:** 1e-4 → 5e-5 (fine-tuning pre-trained model needs lower LR)
- **Epochs:** 8 → 6 (less needed with good initialization)
- **Gate:** WER < 0.45 AND beats served (lowered from 0.50)

### Why This Is Much Better

**Before (v3 approach):**
1. Start: whisper-medium (knows MSA + other languages)
2. Fine-tune: teach it Darija from scratch on 1.5k clips
3. Result: 0.6303 WER (struggles with darija-specific vocabulary)

**After (v6 approach):**
1. Start: whisper-v3-turbo-darija (already knows Darija!)
2. Fine-tune: adapt to Khidmeti domain (plumbing/electrical vocabulary)
3. Expected: <0.40 WER (building on solid darija foundation)

**Analogy:** قبل كنا نعلم طفل ياباني الدارجة، الآن نعلم جزائري متخصص مفردات السباكة!

---

## Data Collection Plan (Optional Enhancement)

النموذج الجديد قد **لا يحتاج 15k clips** لأنه مدرب مسبقاً على الدارجة!

**خطة مرنة:**
1. **أولاً:** نفذ التدريب على 1.5k clips الحالية (Casablanca)
2. **تقييم:** إذا WER < 0.40 → ممتاز، deployment مباشر!
3. **فقط إذا احتجنا:** جمع بيانات إضافية من YouTube/radio

**توقعي:** النموذج الجديد سيحقق نتائج ممتازة حتى بدون بيانات إضافية.

---

## Files Updated

### STT Training Script
- `ml/kaggle/stt_kernel_v6.py` — ⛔ **jamais lancé, ne pas lancer** :
  - ⛔ BASE_MODEL = "anaszil/whisper-large-v3-turbo-darija" (adaptateur PEFT, inchargeable)
  - ⛔ Batch/LR adjusted for larger model
  - ⛔ Gate lowered to WER < 0.45
- `ml/kaggle/stt_kernel_v7b.py` — ✅ ce qui tourne réellement (large-v3 + LoRA r32)

### Documentation
- `ml/STT_V6_CRITICAL_FIX.md` (this file)
- `ml/TRAINING_V6_SUMMARY.md` (original, now superseded)

---

## Next Steps (Priority Order)

### HIGH PRIORITY: Test STT v6 Now!
1. **Push STT v6 to Kaggle** (with correct base model)
2. **Monitor training** (expect 2-3 hours)
3. **Evaluate WER** on Casablanca test set
4. **If WER < 0.45:** Deploy immediately! 🎉
5. **If WER 0.45-0.50:** Still better than v3, consider more data
6. **If WER > 0.50:** Debug (unlikely with darija base)

### MEDIUM PRIORITY: NLU v6
7. Solve dataset size issue for Kaggle push (need to upload as separate dataset)
8. Push NLU training
9. Verify gate passes (prof_acc ≥ 0.90)

### LOW PRIORITY: More Data
10. Only if STT v6 doesn't achieve <0.45 WER
11. Start with YouTube scraping (highest ROI)
12. Process and retrain

---

## Why This Changes Everything

**Before your question:**
- نستخدم whisper-medium الخام (لا يفهم دارجة أصلاً)
- نحاول تعليمه الدارجة من الصفر
- صعب جداً مع 1.5k clips فقط

**After your discovery:**
- نستخدم نموذج **مدرب أصلاً على الدارجة**
- نحسنه فقط لمفردات السباكة/الكهرباء
- أسهل بكثير، نتائج متوقعة أفضل بكثير!

**شكراً على السؤال المهم!** كان سيوفر لنا أسابيع من محاولات جمع بيانات قد لا نحتاجها أصلاً.

---

## Technical Notes

### Model Comparison
| Model | Size | Darija? | Speed | WER (expected) |
|-------|------|---------|-------|----------------|
| whisper-medium (old) | 769M | ❌ No | Fast | 0.63 (actual v3) |
| whisper-large-v3 | 1.5B | ❌ No | Slow | ? |
| **whisper-v3-turbo-darija** | 809M | ✅ **Yes** | **Fast** | **<0.40 (target)** |

Turbo = optimized architecture, almost same speed as medium but better quality.

### Why Not Use v3 Without Fine-tuning?

**نحتاج fine-tuning حتى لو النموذج يفهم الدارجة:**
- المفردات التقنية: سباكة، كهرباء، ميكانيك (domain-specific)
- الأسماء المحلية: بلومبيي، تريسيان، كليمة
- التكامل مع NLU: نفس vocabulary space

لكن الـ fine-tuning الآن **أسهل وأسرع بكثير** لأن الأساس صحيح!

---

## Success Metrics v6

✅ STT base model fixed (darija pre-trained)  
✅ Training hyperparameters adjusted  
✅ Gate lowered to reflect better baseline  
✅ NLU dataset 70k ready (separate issue)  

**Status:** STT v6 ready for immediate training with correct base model!

**Expected timeline:** 2-3 hours training → evaluate → likely ready for production!
