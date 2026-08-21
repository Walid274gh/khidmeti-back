# Khidmeti NLU v6 Training - خطوات التنفيذ

## الحالة الحالية ✅

**تم الإنجاز:**
- ✅ Dataset v6: 70,408 rows (10.7× expansion)
- ✅ Training kernel script جاهز
- ✅ Dataset zip created: `ml/kaggle/nlu_v6_data.zip` (1.0 MB)
- ⛔ STT v6: base model **خاطئ** (anaszil/whisper-large-v3-turbo-darija) — انظر تصحيح 15 أوت في `STT_V6_CRITICAL_FIX.md`

**القرار:** نركز على NLU v6 أولاً، STT الجديد نختبره مباشرة بدون fine-tuning

---

## خطوات التدريب (يدوية - Kaggle API محدود)

### الخطوة 1: رفع Dataset إلى Kaggle

**ملف جاهز:** `/storage/emulated/0/opencode/khid-back/ml/kaggle/nlu_v6_data.zip`

**إجراءات يدوية مطلوبة:**

1. **افتح متصفح وادخل:** https://www.kaggle.com/datasets

2. **اضغط "New Dataset"**

3. **ارفع الملف:**
   - اسحب `nlu_v6_data.zip` من الهاتف
   - أو استخدم زر "Upload Files"

4. **اضبط الإعدادات:**
   - **Title:** `Khidmeti NLU v6 Training Data`
   - **Subtitle:** `70k Algerian Darija rows for intent+profession classification`
   - **Visibility:** Private
   - **License:** CC0-1.0 (Public Domain)

5. **اضغط "Create"**

6. **انسخ رابط Dataset:**
   - سيكون شكله: `https://www.kaggle.com/datasets/YOUR_USERNAME/khidmeti-nlu-v6-data`
   - Dataset slug: `YOUR_USERNAME/khidmeti-nlu-v6-data`

---

### الخطوة 2: تحديث Kernel Script (إذا احتجت)

إذا كان username مختلف عن `walidrbh27khidmeti`:

```bash
# افتح الملف
nano ml/kaggle/build_push_v6.py

# ابحث عن السطر:
DATASET_SLUG = f"{USER}/khidmeti-nlu-v6-data"

# تأكد أن USER صحيح من .env file
```

---

### الخطوة 3: Push Training Kernel

```bash
cd /storage/emulated/0/opencode/khid-back
python3 ml/kaggle/build_push_v6.py push
```

**المتوقع:**
- الكود يرسل kernel script إلى Kaggle
- Kernel يحمل dataset تلقائياً من `/kaggle/input/khidmeti-nlu-v6-data/`
- يبدأ التدريب على T4 GPU

---

### الخطوة 4: مراقبة التدريب

**رابط Kernel:**
```
https://www.kaggle.com/code/YOUR_USERNAME/khidmeti-nlu-train-v6
```

**مراقبة عبر CLI:**
```bash
# حالة التنفيذ
python3 ml/kaggle/build_push_v6.py status

# عرض Logs
python3 ml/kaggle/build_push_v6.py log
```

**مدة متوقعة:** 2-3 ساعات على T4

---

### الخطوة 5: التحقق من النتائج

**Gate للنجاح:**
```
GATE_90_PASS=YES
prof_acc_int8 >= 0.90
```

**إذا نجح:**
- ✅ النموذج يُرفع تلقائياً إلى `Walidrbh27/khidmeti-nlu` على HuggingFace
- ✅ Metadata يحتوي على accuracy metrics
- ✅ جاهز للنشر في ai-nlu container

**إذا فشل (prof_acc < 0.90):**
- ❌ لا يُرفع النموذج
- 📊 راجع per-profession breakdown في logs
- 🔧 قد نحتاج augmentation إضافي للمهن الضعيفة

---

## ملفات تم إنشاؤها

### Scripts
- `ml/kaggle/prepare_dataset.py` - إنشاء dataset zip
- `ml/kaggle/build_push_v6.py` - push kernel إلى Kaggle
- `ml/kaggle/nlu_v6_data.zip` - dataset للرفع اليدوي

### Dataset Files (داخل zip)
- `synth_v6_final.csv` - 70,408 training rows
- `eval_heldout.csv` - evaluation set (unchanged)
- `labels.json` - intents + professions
- `dataset-metadata.json` - Kaggle metadata

### Documentation
- `ml/DEPLOYMENT_INSTRUCTIONS.md` - هذا الملف

---

## بعد نجاح التدريب

### تحديث ai-nlu Container

```bash
cd /storage/emulated/0/opencode/khid-back/docker/ai-nlu

# النموذج الجديد سيُحمّل تلقائياً من HF
# عند إعادة تشغيل الـ container:
docker-compose restart ai-nlu

# أو إعادة بناء:
docker-compose up -d --build ai-nlu
```

### ⛔ اختبار STT الجديد (بدون fine-tuning) — ملغى، لا تنفّذه

**السبب:** `anaszil/whisper-large-v3-turbo-darija` هو **adaptateur PEFT/LoRA**، و
faster-whisper لا يستطيع تحميله؛ وضعه في `STT_MODEL` يضع الحاوية في FALLBACK صامت.
كما أنه ليس دارجة جزائرية. تصحيح كامل بالأرقام المقيسة في `STT_V6_CRITICAL_FIX.md`.

**ما يجب فعله بدلاً منه:** أبقِ `STT_MODEL=Walidrbh27/khidmeti-stt` (أوزاننا CT2)،
وحسّنها بـ `ml/kaggle/stt_kernel_v7b.py` (socle `openai/whisper-large-v3` + LoRA).

<details><summary>النص الأصلي الخاطئ (للأرشيف)</summary>

**النموذج الجديد:** `anaszil/whisper-large-v3-turbo-darija`

**طريقة الاختبار:**

1. **تحديث ai-stt container لاستخدام النموذج الجديد:**

```python
# في docker/ai-stt/app.py أو ملف التكوين
MODEL_NAME = "anaszil/whisper-large-v3-turbo-darija"
```

</details>

2. **إعادة التشغيل:**
```bash
docker-compose restart ai-stt
```

3. **اختبار عملي:**
   - سجل صوت: "نحتاج بلومبيي في الدار"
   - تحقق من النص المُستخرج
   - قارن مع v3 (whisper-medium)

4. **القياس:**
   - إذا التعرف أفضل: ✅ نشر مباشر!
   - إذا نفس المستوى أو أسوأ: 🔧 نحتاج fine-tuning (stt_kernel_v6.py جاهز)

---

## Troubleshooting

### مشكلة: Kaggle API 403 Permission Denied
**الحل:** Kaggle API لا يدعم رفع datasets برمجياً - يجب الرفع اليدوي عبر UI

### مشكلة: Kernel script >1MB
**الحل:** ✅ محلول - نستخدم Kaggle dataset بدلاً من inline data

### مشكلة: Dataset slug خاطئ
**الحل:** تأكد من username في `.env` يطابق Kaggle username

### مشكلة: Gate فشل (prof_acc < 0.90)
**الحلول:**
1. راجع per-profession accuracy في logs
2. أضف hand-curated examples للمهن الضعيفة
3. زد augmentation لتلك المهن
4. أعد التدريب

---

## النتيجة المتوقعة

**NLU v6:**
- v5: 88.72% accuracy (6.5k rows)
- v6: ≥90% accuracy (70k rows) - **هدف**

**STT (بدون fine-tuning):**
- v3: 0.6303 WER (whisper-medium خام + fine-tuning)
- v6 test: <?? WER (whisper-v3-turbo-darija مباشرة)
- متوقع: <0.50 WER (لأن النموذج مدرب على الدارجة)

---

## الخلاصة

**جاهز للتنفيذ:**
1. ✅ Dataset zip موجود: `nlu_v6_data.zip`
2. 📤 ارفع يدوياً إلى Kaggle
3. 🚀 Run: `python3 ml/kaggle/build_push_v6.py push`
4. ⏱️ انتظر 2-3 ساعات
5. 🎉 تحقق من النتيجة + نشر

**ملاحظة:** كل الكود جاهز، الخطوة الوحيدة اليدوية هي رفع dataset zip إلى Kaggle UI.
