# KHIDMETI — Deploy Kit (سحابة مجانية 100%، بلا بطاقة)

**المبدأ:** صفر تعديل على الكود. وضع local/Codespace (`make start` + compose) يبقى كما هو تماماً — الـAPI يقرأ عناوين خدمات الـAI من متغيرات البيئة أصلاً، فالفرق بين local والسحابة هو **قيم المتغيرات فقط**:

| الخدمة | local (compose) | سحابة |
|---|---|---|
| API + Socket.IO | حاوية api | Render Free |
| ai-nlu / stt / vision / embed | حاويات :8013–:8015/:8012 | 4 تطبيقات Modal |
| MongoDB / Redis / Qdrant | حاويات | Atlas M0 / Upstash / Qdrant Cloud |

هذا المجلد إضافي بالكامل: `render.yaml` + `deploy/modal/*` + `deploy/scripts/run-seeds.sh`.

---

## 1. الحسابات المطلوبة

| المنصة | ماذا نأخذ | الحالة |
|---|---|---|
| MongoDB Atlas | `MONGODB_URI` (cluster M0، مستخدم db، Network Access = 0.0.0.0/0) | 🔄 جديد |
| Upstash | `REDIS_URL` (Redis Cloud، TLS) | 🔄 جديد |
| Qdrant Cloud | `QDRANT_URL` + `QDRANT_API_KEY` (free 1GB) | 🔄 جديد |
| Cloudinary | `CLOUDINARY_*` الثلاثة | 🔄 جديد |
| GitHub | repo خاص يرفع فيه khid-back | 🔄 جديد |
| Render | Web Service من الـblueprint | 🔄 جديد |
| Modal | حساب + Secret واحد للـHF_TOKEN | 🔄 جديد |
| UptimeRobot | monitor على `/health` | 🔄 جديد |
| Firebase / HF_TOKEN / MapTiler | — | ✅ موجودة في `.env.cloud` |

⚠️ عند نسخ `REDIS_URL` و`QDRANT_API_KEY` من متصفح: راقب **المسافات** — النسخة القديمة في `.env.cloud` كانت فيها مسافات مدمجة تكسر الاتصال.

## 2. Modal (النماذج الأربعة)

```bash
pip install modal && modal setup          # مرة واحدة
modal secret create khidmeti-hf HF_TOKEN=hf_xxx   # نفس التوكن الصالح
# من جذر khid-back:
modal deploy deploy/modal/nlu_app.py
modal deploy deploy/modal/stt_app.py
modal deploy deploy/modal/vision_app.py
modal deploy deploy/modal/embed_app.py
```

كل app يعيد رابطاً مثل `https://<workspace>--khidmeti-nlu-serve.modal.run`. انسخها. التطبيقات تشغّل `docker/ai-*/server.py` **بنفس الكود حرفياً** — نفس العقود، نفس meta.json كمصدر حقيقة لمعاملات LM، والأوزان تنزل من مستودعاتك الخاصة إلى Volume دائم فلا يُعاد تنزيلها.

## 3. GitHub + Render

```bash
cd khid-back && git init && git add -A && git commit -m "deploy kit"
gh repo create khidmeti-back --private --source=. --push   # .gitignore يحجب .env* وdocker/models أصلاً
```
Render → New → Blueprint → اختر الـrepo → سيقرأ `deploy/render.yaml` (Root Directory للـblueprint = `deploy`). املأ قيم `sync:false` من الجدول أدناه.

## 4. متغيرات Render (المرجع الكامل)

```bash
NODE_ENV=production
NODE_OPTIONS=--max-old-space-size=384      # instance free = 512MB
FLYWHEEL_DIR=/tmp/flywheel                 # قرص free مؤقت — بيانات الموافقة فقط
API_BASE_URL=https://khidmeti-api.onrender.com
CORS_ORIGINS=https://khidmeti.pages.dev,https://khidmeti-api.onrender.com,http://localhost:8080
EMBEDDINGS_MODEL=nomic-embed-text-v1.5

MONGODB_URI=mongodb+srv://...              # 🔄 Atlas
REDIS_URL=rediss://...                     # 🔄 Upstash — بلا مسافات!
QDRANT_URL=https://...qdrant.io:6333       # 🔄
QDRANT_API_KEY=...                         # 🔄
CLOUDINARY_CLOUD_NAME=...                  # 🔄
CLOUDINARY_API_KEY=...                     # 🔄
CLOUDINARY_API_SECRET=...                  # 🔄
MAPTILER_API_KEY=<من .env.cloud>           # ✅
FIREBASE_PROJECT_ID=<...>                  # ✅ الثلاثة من .env.cloud
FIREBASE_CLIENT_EMAIL=<...>
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n..."
FIELD_ENC_KEY=$(openssl rand -hex 32)      # 🔑 جديد — قاعدة جديدة؛ احفظه في مكان آمن، فقدانه = فقدان كل PII مشفر
FIELD_ENC_PEPPER=$(openssl rand -hex 32)

NLU_URL=https://<ws>--khidmeti-nlu-serve.modal.run
STT_URL=https://<ws>--khidmeti-stt-serve.modal.run
VISION_URL=https://<ws>--khidmeti-vision-serve.modal.run
EMBEDDINGS_URL=https://<ws>--khidmeti-embed-serve.modal.run/v1   # ينتهي بـ/v1
```

غير المطلوب على Render: `HF_TOKEN` (النماذج على Modal)، `STT_*`/`VISION_BIAS`/`VISION_REPO` (مدمجة داخل تطبيقات Modal)، `NGROK_*`/`MACHINE_HOST`/`COMPOSE_FILE` (أصبحت بلا معنى).

## 5. Seeds (قاعدة فارغة)

```bash
export MONGODB_URI="mongodb+srv://..."
./deploy/scripts/run-seeds.sh     # professions ثم workers — idempotent
```

## 6. UptimeRobot

Monitor HTTPS على `https://khidmeti-api.onrender.com/health` كل 10 دقائق — يمنع نوم Render (ينام بعد 15 دقيقة خمول، ويستيقظ خلال ~دقيقة).

## 7. Flutter

بدّل `api_base_url` إلى عنوان Render وأعد البناء في FlutLab. (لاحقاً: Firebase Remote Config يجعل التبديل بلا APK جديد.)

## 8. اختبار دخان

```bash
curl https://khidmeti-api.onrender.com/health
curl -X POST $NLU_URL/classify  -H 'Content-Type: application/json' -d '{"text":"محتاج سباك في وهران"}'
curl -X POST $VISION_URL/classify -H 'Content-Type: application/octet-stream' --data-binary @test.jpg
curl -X POST $STT_URL/transcribe  -H 'Content-Type: audio/ogg' --data-binary @test.ogg
curl -X POST $EMBEDDINGS_URL/embeddings -H 'Content-Type: application/json' -d '{"model":"nomic-embed-text-v1.5","input":"سباك"}'
```

---

### ملاحظات صادقة

- **Cold starts في Modal**: بعد 5 دقائق خمول تنام الحاوية؛ أول ندوة قد تستغرق 10–60 ثانية. NLU مهلته 8 ثوانٍ فقط → أول طلب بعد خمول طويل يسقط بأمان إلى FALLBACK اليدوي مرة واحدة ثم ينجو. STT مهلتها 120 ثانية فلا تتأثر.
- **التكلفة**: scale-to-zero + نافذة سخونة 300 ثانية ≈ بضعة دولارات شهرياً من أصل $30 المجانية.
- **flywheel على Render مؤقت** (`/tmp`) — يُمسح مع كل نشر. مقبول لبيانات الموافقة الاختيارية؛ التخزين الدائم يتطلب disk مدفوعاً أو Cloudinary.
