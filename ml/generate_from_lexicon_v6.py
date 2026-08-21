#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
# ml/generate_from_lexicon_v6.py — 65k+ rows via exhaustive lexicon combinations
#
# Strategy: Generate directly from LEX (profession names × problems × templates)
# instead of relying only on pre-generated v5. This unlocks the full combinatorial
# space: 16 professions × ~20 problems each × ~50 templates × modifiers = 65k+
# ══════════════════════════════════════════════════════════════════════════════
import csv
import json
import random
from pathlib import Path
from collections import Counter

random.seed(20260810)

OUT = Path(__file__).resolve().parent / 'dataset'
LABELS = json.loads((OUT / 'labels.json').read_text())

# Import full LEX from generate_dataset.py
import sys
sys.path.insert(0, str(OUT.parent))
from generate_dataset import LEX, WILAYAS

# Extended templates (beyond original generate_dataset.py)
TEMPLATES_FIND_WORKER_AR = [
    'راني نحوس على {name} في {wilaya}',
    'نحتاج {name} مليح في {wilaya}',
    'واش كاين شي {name} قريب من {wilaya}؟',
    'دلوني على {name} شاطر ومعقول',
    'عندي {problem}',
    '{problem}',
    '{problem} واش نديار',
    '{problem} عاونوني',
    'نقدر نلقى {name} يجي اليوم؟',
    'شكون عندو نيميرو تاع {name}',
    '{name} في منطقة {wilaya}',
    'نبحث على {name} ضروري',
    'محتاج {name} بسرعة',
    '{problem} ونحتاج واحد يصلحهالي',
    '{problem} شكون يعرف {name}',
    'واش عندك {name} يجي للدار',
    '{name} قريب مني في {wilaya}',
    'عندي مشكل: {problem}',
    '{problem} وخلاص',
    'الله يرحم الوالدين {name} في {wilaya}',
    '{problem} كارثة',
    '{problem} يخوف',
    '{problem} ما نقدرش نبات هكذا',
    'نحتاج {name} اليوم أو غدوة',
    '{problem} قبل العيد',
    '{problem} نهار الجمعة',
    '{problem} هاذ الأسبوع',
    '{problem} في الدار',
    'مشكلة في الدار {problem}',
    '{problem} والدار كلها',
]

TEMPLATES_FIND_WORKER_LAT = [
    'n7awes 3la {name} f {wilaya}',
    'n7taj {name} mli7 f {wilaya}',
    'wach kayn chi {name} 9rib men {wilaya}',
    'dllouni 3la {name} chater w m3a9oul',
    '3andi {problem}',
    '{problem}',
    '{problem} wach ndir',
    '{problem} 3awnouni',
    'n9der nel9a {name} yji lyouma',
    'chkoun 3andou numero ta3 {name}',
    '{name} f {wilaya}',
    'bghit {name} darori',
    'm7taj {name} bsor3a',
    '{problem} w n7taj wa7ed yssal7hali',
    '{problem} chkoun y3aref {name}',
    'wach 3andek {name} yji l dar',
    '{name} 9rib menni f {wilaya}',
    '3andi mochkil {problem}',
    '{problem} w khalas',
    'allah yer7em waldine {name} f {wilaya}',
    '{problem} catastrophe',
    '{problem} ykhawwef',
    '{problem} ma n9derch nebbet hakda',
    'n7taj {name} lyouma wla ghodwa',
    '{problem} 9bel l3id',
    '{problem} nhar jem3a',
    '{problem} had semana',
    '{problem} f dar',
    'mochkla f dar {problem}',
    '{problem} w dar kamla',
]

TEMPLATES_URGENT_AR = [
    'عاجل {problem}',
    'مستعجل {problem}',
    'ضروري {problem}',
    'توا توا {problem}',
    '{problem} بسرعة',
    '{problem} عافاكم',
    '{problem} الله يرحم الوالدين',
    'نحتاج {name} بسرعة {problem}',
    '{problem} ما يتأخرش',
]

TEMPLATES_URGENT_LAT = [
    'urgent {problem}',
    'darori {problem}',
    'bsor3a {problem}',
    '{problem} vite',
    '{problem} 3afakom',
    '{problem} allah yer7em waldine',
    'n7taj {name} bsor3a {problem}',
    '{problem} ma yetakhharech',
]

TEMPLATES_PRICE_AR = [
    'بشحال {name}',
    'شحال تكلفة {name}',
    'شحال الثمن تاع {problem}',
    'كم يكلف {name}',
    'واش غالي {name}',
    'عطوني فكرة على الثمن تاع {problem}',
    'شحال يدي {name} باش {problem}',
]

TEMPLATES_PRICE_LAT = [
    'bch7al {name}',
    'ch7al coute {name}',
    'ch7al thaman ta3 {problem}',
    'prix ta3 {name}',
    'wach ghali {name}',
    '3tiwni idée 3la thaman ta3 {problem}',
    'ch7al ydir {name} bach {problem}',
]

def generate_from_profession(prof_key: str, prof_data: dict) -> list:
    """Generate all combinations for one profession."""
    rows = []

    # Combine all name variants
    all_names_ar = prof_data.get('ar_names', [])
    all_names_lat = prof_data.get('lat_names', [])

    # Combine all problem variants
    all_problems_ar = prof_data.get('ar_problems', [])
    all_problems_lat = prof_data.get('lat_problems', [])

    # Generate find_worker (ar)
    for template in TEMPLATES_FIND_WORKER_AR:
        # Name + wilaya variations
        if '{name}' in template and '{wilaya}' in template:
            for name in all_names_ar:
                for wilaya_ar, _ in WILAYAS[:10]:  # Sample wilayas
                    text = template.replace('{name}', name).replace('{wilaya}', wilaya_ar)
                    rows.append({'text': text, 'intent': 'find_worker', 'profession': prof_key})

        # Name only
        elif '{name}' in template and '{problem}' not in template:
            for name in all_names_ar:
                text = template.replace('{name}', name)
                if '{wilaya}' in text:
                    for wilaya_ar, _ in WILAYAS[:10]:
                        text2 = text.replace('{wilaya}', wilaya_ar)
                        rows.append({'text': text2, 'intent': 'find_worker', 'profession': prof_key})
                else:
                    rows.append({'text': text, 'intent': 'find_worker', 'profession': prof_key})

        # Problem only
        elif '{problem}' in template:
            for problem in all_problems_ar:
                text = template.replace('{problem}', problem)
                if '{name}' in text:
                    for name in all_names_ar:
                        text2 = text.replace('{name}', name)
                        rows.append({'text': text2, 'intent': 'find_worker', 'profession': prof_key})
                else:
                    rows.append({'text': text, 'intent': 'find_worker', 'profession': prof_key})

    # Generate find_worker (lat)
    for template in TEMPLATES_FIND_WORKER_LAT:
        if '{name}' in template and '{wilaya}' in template:
            for name in all_names_lat:
                for _, wilaya_lat in WILAYAS[:10]:
                    text = template.replace('{name}', name).replace('{wilaya}', wilaya_lat)
                    rows.append({'text': text, 'intent': 'find_worker', 'profession': prof_key})

        elif '{name}' in template and '{problem}' not in template:
            for name in all_names_lat:
                text = template.replace('{name}', name)
                if '{wilaya}' in text:
                    for _, wilaya_lat in WILAYAS[:10]:
                        text2 = text.replace('{wilaya}', wilaya_lat)
                        rows.append({'text': text2, 'intent': 'find_worker', 'profession': prof_key})
                else:
                    rows.append({'text': text, 'intent': 'find_worker', 'profession': prof_key})

        elif '{problem}' in template:
            for problem in all_problems_lat:
                text = template.replace('{problem}', problem)
                if '{name}' in text:
                    for name in all_names_lat:
                        text2 = text.replace('{name}', name)
                        rows.append({'text': text2, 'intent': 'find_worker', 'profession': prof_key})
                else:
                    rows.append({'text': text, 'intent': 'find_worker', 'profession': prof_key})

    # Generate urgent_service (ar)
    for template in TEMPLATES_URGENT_AR:
        if '{problem}' in template:
            for problem in all_problems_ar[:10]:  # Sample problems
                text = template.replace('{problem}', problem)
                if '{name}' in text:
                    for name in all_names_ar:
                        text2 = text.replace('{name}', name)
                        rows.append({'text': text2, 'intent': 'urgent_service', 'profession': prof_key})
                else:
                    rows.append({'text': text, 'intent': 'urgent_service', 'profession': prof_key})

    # Generate urgent_service (lat)
    for template in TEMPLATES_URGENT_LAT:
        if '{problem}' in template:
            for problem in all_problems_lat[:10]:
                text = template.replace('{problem}', problem)
                if '{name}' in text:
                    for name in all_names_lat:
                        text2 = text.replace('{name}', name)
                        rows.append({'text': text2, 'intent': 'urgent_service', 'profession': prof_key})
                else:
                    rows.append({'text': text, 'intent': 'urgent_service', 'profession': prof_key})

    # Generate price_inquiry (ar)
    for template in TEMPLATES_PRICE_AR:
        if '{name}' in template:
            for name in all_names_ar:
                text = template.replace('{name}', name)
                if '{problem}' in text:
                    for problem in all_problems_ar[:5]:
                        text2 = text.replace('{problem}', problem)
                        rows.append({'text': text2, 'intent': 'price_inquiry', 'profession': prof_key})
                else:
                    rows.append({'text': text, 'intent': 'price_inquiry', 'profession': prof_key})
        elif '{problem}' in template:
            for problem in all_problems_ar[:5]:
                text = template.replace('{problem}', problem)
                rows.append({'text': text, 'intent': 'price_inquiry', 'profession': prof_key})

    # Generate price_inquiry (lat)
    for template in TEMPLATES_PRICE_LAT:
        if '{name}' in template:
            for name in all_names_lat:
                text = template.replace('{name}', name)
                if '{problem}' in text:
                    for problem in all_problems_lat[:5]:
                        text2 = text.replace('{problem}', problem)
                        rows.append({'text': text2, 'intent': 'price_inquiry', 'profession': prof_key})
                else:
                    rows.append({'text': text, 'intent': 'price_inquiry', 'profession': prof_key})
        elif '{problem}' in template:
            for problem in all_problems_lat[:5]:
                text = template.replace('{problem}', problem)
                rows.append({'text': text, 'intent': 'price_inquiry', 'profession': prof_key})

    return rows

def generate_oos_greeting() -> list:
    """Generate out-of-scope and greeting samples."""
    rows = []

    # Greetings
    greetings = [
        'سلام', 'كيف راك', 'صباح الخير', 'مساء الخير', 'وش راك',
        'لاباس', 'قداش راك', 'نهارك مبروك', 'cv', 'هلا',
        'salam', 'ki rak', 'sba7 el khir', 'labas', '3lach',
        'bonjour', 'bonsoir', 'ca va', 'wech rak', 'kifech',
    ]
    for g in greetings:
        rows.append({'text': g, 'intent': 'greeting_chitchat', 'profession': 'none'})

    # Chitchat
    chitchat = [
        'شكرا بزاف', 'الله يحفظك', 'مرسي', 'بارك الله فيك',
        'ربي يسترك', 'معليش', 'انشاء الله', 'ماشي مشكل',
        'merci', 'baraka llah fik', 'machi mochkil', 'inchallah',
        'nchalah', 'rabi yester', 'allah y7afdek', 'cv hamdullah',
    ]
    for c in chitchat:
        rows.append({'text': c, 'intent': 'greeting_chitchat', 'profession': 'none'})

    # OOS (out of scope)
    oos = [
        'نحب نشري تلفون', 'كاين طاكسي', 'نروح للطبيب', 'نحجز فندق',
        'واش عندك فيزا', 'نحب ندير باسبور', 'مطعم قريب مني',
        'وين نلقى محامي', 'نحب نسجل في الجامعة', 'كراء طوموبيل',
        'bghit nechri telephone', 'kayn taxi', 'nrouh l tbi b',
        'reservation hotel', 'wach 3andek visa', 'restaurant 9rib',
        'avocat mlih', 'inscription universite', 'location voiture',
    ]
    for o in oos:
        rows.append({'text': o, 'intent': 'out_of_scope', 'profession': 'none'})

    # App questions
    app_q = [
        'كيفاش نخلص', 'واش مجاني', 'شحال الوقت تاع الدليفري',
        'كيفاش نسجل', 'وين نلقى الملف تاعي', 'واش كاين تطبيق',
        'kifach nkhalles', 'wach mejani', 'ch7al wa9t delivery',
        'kifach nregistri', 'win nel9a profil', 'wach kayn app',
        'comment je paye', 'c est gratuit', 'combien de temps',
    ]
    for a in app_q:
        rows.append({'text': a, 'intent': 'app_question', 'profession': 'none'})

    return rows

def main():
    print("=" * 80)
    print("Exhaustive Lexicon-Based Generation — Target 65k+")
    print("=" * 80)

    all_rows = []

    # Generate from each profession
    for prof_key, prof_data in LEX.items():
        print(f"Generating for {prof_key}...")
        prof_rows = generate_from_profession(prof_key, prof_data)
        all_rows.extend(prof_rows)
        print(f"  Generated {len(prof_rows)} rows")

    # Add OOS/greetings
    print("Generating OOS/greetings...")
    oos_rows = generate_oos_greeting()
    all_rows.extend(oos_rows)
    print(f"  Generated {len(oos_rows)} rows")

    print(f"\nTotal before dedup: {len(all_rows)}")

    # Dedup
    seen = set()
    deduped = []
    for row in all_rows:
        key = (row['text'].strip().lower(), row['intent'], row['profession'])
        if key not in seen and 5 <= len(row['text']) <= 250:
            seen.add(key)
            row['source'] = 'synth_v6_lexicon'
            deduped.append(row)

    print(f"After dedup: {len(deduped)}")

    # Save
    out_path = OUT / 'synth_v6_lexicon.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'intent', 'profession', 'source'])
        writer.writeheader()
        writer.writerows(deduped)

    print(f"\nSaved to {out_path}")
    print(f"Intent breakdown: {Counter(r['intent'] for r in deduped)}")
    print(f"Profession breakdown: {Counter(r['profession'] for r in deduped)}")

if __name__ == '__main__':
    main()
