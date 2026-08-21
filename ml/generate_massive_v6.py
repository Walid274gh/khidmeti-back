#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
# ml/generate_massive_v6.py — 65k+ NLU rows via deep template expansion
#
# Strategy: multiply base lexicon with contextual dimensions:
#   - User personas (elderly, young couple, landlord, tenant, business)
#   - Urgency levels (emergency, urgent, scheduled, preventive)
#   - Time contexts (before event, seasonal, after purchase, recurring)
#   - Problem combinations (multiple issues, cascading failures)
#   - Quality modifiers (cheap, professional, guaranteed, fast)
#   - Location specificity (neighborhood, building type, access constraints)
#
# Each dimension multiplies existing templates geometrically.
# Target: 6.5k base × 10 dimensions = 65k+ unique realistic rows.
# ══════════════════════════════════════════════════════════════════════════════
import csv
import json
import random
import re
from pathlib import Path
from collections import Counter

random.seed(20260810)

OUT = Path(__file__).resolve().parent / 'dataset'
LABELS = json.loads((OUT / 'labels.json').read_text())

# Load base v5 as foundation
v5_rows = list(csv.DictReader(open(OUT / 'synth_v5.csv')))
print(f"Base v5: {len(v5_rows)} rows")

# ═══════════════════════════════════════════════════════════════════════════
# Contextual multipliers (each adds realistic variation)
# ═══════════════════════════════════════════════════════════════════════════

URGENCY_PREFIXES = {
    'ar': [
        'عاجل: ', 'توا توا: ', 'ضروري: ', 'مستعجل: ', 'عافاكم: ',
        'بليز: ', 'الله يرحم الوالدين: ', 'نحتاج بسرعة: ',
    ],
    'lat': [
        'urgent: ', 'ya kho: ', 'stp: ', 'bghit bezzar: ', 'darori: ',
        'besoin vite: ', 'yji daba: ', 's il vous plait: ',
    ],
}

TIME_CONTEXT_AR = [
    'قبل العيد بيومين', 'نهار الجمعة', 'الأسبوع الجاي', 'اليوم أو غدوة',
    'قبل رمضان', 'قبل الشتا', 'قبل الصيف', 'نهار السبت الصباح',
    'نهار الأحد العشية', 'هاذ الشهر', 'الأسبوع هذا',
]

TIME_CONTEXT_LAT = [
    'demain matin', 'vendredi', 'la semaine prochaine', 'aujourd hui',
    'avant ramadan', 'avant l ete', 'samedi', 'dimanche aprem',
    'ce mois', 'cette semaine',
]

USER_PERSONAS_AR = [
    'راني مستأجر في', 'عندي دار جديدة في', 'راني مالك عمارة في',
    'عندي محل في', 'والدة مريضة في الدار في', 'راني طالب في',
    'عندي فيلا في', 'راني عجوز وحداني في', 'عندنا شركة في',
]

USER_PERSONAS_LAT = [
    'rani mosta2jer f', 'dar jdida f', 'malk 3amara f',
    '3andi magazin f', 'walida marida f', 'villa f',
    '3ajouza wa7dani f', '3andna societe f', 'appartement f',
]

QUALITY_MODIFIERS_AR = [
    'رخيص', 'محترف', 'معقول', 'مضمون', 'ثقة', 'ناشط',
    'سريع', 'بالضمان', 'مجرب', 'ياسر مليح', 'عندو سمعة',
]

QUALITY_MODIFIERS_LAT = [
    'pas cher', 'professionnel', 'm3a9oul', 'madmoun', 'serieux',
    'sari3', 'm3a garantie', 'mjarreb', 'ye3raf khdemtou', 'bon prix',
]

PROBLEM_INTENSIFIERS_AR = [
    'كارثة', 'ما عادش يتحمل', 'خايف يولي أكثر', 'راه خطير',
    'يزيد كل يوم', 'ما نقدرش نبات هكذا', 'خلاص طفح الكيل',
]

PROBLEM_INTENSIFIERS_LAT = [
    'catastrophe', 'ma 3adch yet7amel', 'khayef ywelli akther',
    'khtar bezzaf', 'yzid kol yum', 'ma n9derch nebbet hakda',
]

LOCATION_SPECIFICS_AR = [
    'في الطابق الخامس', 'في عمارة قديمة', 'في فيلا',
    'في بيت شعبي', 'في حي راقي', 'في المدينة الجامعية',
    'في المركز', 'في الضواحي', 'في منطقة بعيدة',
]

LOCATION_SPECIFICS_LAT = [
    'fel 5eme etage', 'f 3amara 9dima', 'f villa',
    'f dar cha3biya', 'f 7ay ra9i', 'f centre ville',
    'f dawahi', 'f manti9a ba3ida', 'f appartement sghir',
]

# Multi-problem templates (cascading failures)
MULTI_PROBLEM_CONNECTORS_AR = [
    ' و ', ' زيد ', ' وكيما ', ' وباراكة ', ' من فوق ',
]

MULTI_PROBLEM_CONNECTORS_LAT = [
    ' w ', ' zid ', ' w kima ', ' w baraka ', ' w men fou9 ',
]

SEASONAL_CONTEXTS_AR = [
    'مع الحر الشديد هذا', 'مع البرد الكبير', 'من الأمطار',
    'من الرطوبة', 'من الصيف', 'من الشتا الباردة',
]

SEASONAL_CONTEXTS_LAT = [
    'm3a l7ar chadid', 'm3a l bard', 'men les pluies',
    'men l humidité', 'men l ete', 'm3a chta barda',
]

# ═══════════════════════════════════════════════════════════════════════════
# Generation strategies
# ═══════════════════════════════════════════════════════════════════════════

def is_arabic_script(text: str) -> bool:
    """Check if text contains Arabic script."""
    return any('؀' <= c <= 'ۿ' for c in text)

def add_urgency(text: str, prob=0.15) -> str:
    """Prefix with urgency marker."""
    if random.random() > prob:
        return text
    prefixes = URGENCY_PREFIXES['ar' if is_arabic_script(text) else 'lat']
    return random.choice(prefixes) + text

def add_time_context(text: str, prob=0.20) -> str:
    """Append time constraint."""
    if random.random() > prob:
        return text
    times = TIME_CONTEXT_AR if is_arabic_script(text) else TIME_CONTEXT_LAT
    return text + ' ' + random.choice(times)

def add_persona(text: str, prob=0.12) -> str:
    """Prefix with user persona."""
    if random.random() > prob:
        return text
    personas = USER_PERSONAS_AR if is_arabic_script(text) else USER_PERSONAS_LAT
    # Insert after first verb/noun
    words = text.split()
    if len(words) > 3:
        insert_pos = random.randint(1, min(3, len(words)-1))
        words.insert(insert_pos, random.choice(personas))
        return ' '.join(words)
    return random.choice(personas) + ' ' + text

def add_quality_modifier(text: str, prob=0.18) -> str:
    """Append quality requirement."""
    if random.random() > prob:
        return text
    mods = QUALITY_MODIFIERS_AR if is_arabic_script(text) else QUALITY_MODIFIERS_LAT
    return text + ' ' + random.choice(mods)

def add_problem_intensifier(text: str, prob=0.10) -> str:
    """Add dramatic intensification."""
    if random.random() > prob:
        return text
    ints = PROBLEM_INTENSIFIERS_AR if is_arabic_script(text) else PROBLEM_INTENSIFIERS_LAT
    return text + ' ' + random.choice(ints)

def add_location_specific(text: str, prob=0.15) -> str:
    """Add location constraint."""
    if random.random() > prob:
        return text
    locs = LOCATION_SPECIFICS_AR if is_arabic_script(text) else LOCATION_SPECIFICS_LAT
    return text + ' ' + random.choice(locs)

def add_seasonal_context(text: str, prob=0.08) -> str:
    """Add seasonal explanation."""
    if random.random() > prob:
        return text
    seasons = SEASONAL_CONTEXTS_AR if is_arabic_script(text) else SEASONAL_CONTEXTS_LAT
    return text + ' ' + random.choice(seasons)

def create_multi_problem(row1, row2) -> dict:
    """Combine two problems into one request (if same profession)."""
    if row1['profession'] != row2['profession']:
        return None
    if row1['profession'] == 'none' or row1['intent'] != 'find_worker':
        return None

    # Only combine if both are problems (not greetings/questions)
    connector = (MULTI_PROBLEM_CONNECTORS_AR if is_arabic_script(row1['text'])
                 else MULTI_PROBLEM_CONNECTORS_LAT)

    combined = row1['text'] + random.choice(connector) + row2['text']
    return {
        'text': combined,
        'intent': 'find_worker',
        'profession': row1['profession'],
        'source': 'synth_v6_multi_problem'
    }

# ═══════════════════════════════════════════════════════════════════════════
# Multiplication pipeline
# ═══════════════════════════════════════════════════════════════════════════

def multiply_contexts(base_rows: list) -> list:
    """Apply all contextual multipliers to base rows."""
    expanded = []

    # Pass 1: Base rows + single-dimension variations
    for row in base_rows:
        text = row['text']
        intent = row['intent']
        profession = row['profession']

        # Original
        expanded.append({
            'text': text,
            'intent': intent,
            'profession': profession,
            'source': 'synth_v5_base'
        })

        # Skip non-service requests for most variations
        skip_variations = intent not in ['find_worker', 'urgent_service', 'price_inquiry']

        if not skip_variations:
            # Urgency
            if random.random() < 0.3:
                expanded.append({
                    'text': add_urgency(text, prob=1.0),
                    'intent': 'urgent_service' if intent == 'find_worker' else intent,
                    'profession': profession,
                    'source': 'synth_v6_urgency'
                })

            # Time context
            if random.random() < 0.35:
                expanded.append({
                    'text': add_time_context(text, prob=1.0),
                    'intent': intent,
                    'profession': profession,
                    'source': 'synth_v6_time'
                })

            # Persona
            if random.random() < 0.25:
                expanded.append({
                    'text': add_persona(text, prob=1.0),
                    'intent': intent,
                    'profession': profession,
                    'source': 'synth_v6_persona'
                })

            # Quality
            if random.random() < 0.30:
                expanded.append({
                    'text': add_quality_modifier(text, prob=1.0),
                    'intent': intent,
                    'profession': profession,
                    'source': 'synth_v6_quality'
                })

            # Location
            if random.random() < 0.28:
                expanded.append({
                    'text': add_location_specific(text, prob=1.0),
                    'intent': intent,
                    'profession': profession,
                    'source': 'synth_v6_location'
                })

            # Intensifier
            if random.random() < 0.20:
                expanded.append({
                    'text': add_problem_intensifier(text, prob=1.0),
                    'intent': intent,
                    'profession': profession,
                    'source': 'synth_v6_intensifier'
                })

            # Seasonal
            if random.random() < 0.15:
                expanded.append({
                    'text': add_seasonal_context(text, prob=1.0),
                    'intent': intent,
                    'profession': profession,
                    'source': 'synth_v6_seasonal'
                })

    print(f"After single-dimension: {len(expanded)} rows")

    # Pass 2: Multi-dimension combinations (2-3 modifiers)
    base_count = len(expanded)
    for i in range(min(len(base_rows), 3000)):  # Sample for multi-dim
        row = random.choice(base_rows[:base_count])
        if row['intent'] not in ['find_worker', 'urgent_service']:
            continue

        text = row['text']
        # Apply 2-3 random modifiers
        modifiers = random.sample([
            add_urgency, add_time_context, add_quality_modifier,
            add_location_specific, add_problem_intensifier
        ], k=random.randint(2, 3))

        for mod in modifiers:
            text = mod(text, prob=0.8)

        expanded.append({
            'text': text,
            'intent': 'urgent_service' if 'urgent' in str(modifiers) else row['intent'],
            'profession': row['profession'],
            'source': 'synth_v6_multi_dim'
        })

    print(f"After multi-dimension: {len(expanded)} rows")

    # Pass 3: Multi-problem combinations
    service_rows = [r for r in base_rows if r['intent'] == 'find_worker' and r['profession'] != 'none']
    for _ in range(min(len(service_rows), 1500)):
        r1, r2 = random.sample(service_rows, 2)
        multi = create_multi_problem(r1, r2)
        if multi:
            expanded.append(multi)

    print(f"After multi-problem: {len(expanded)} rows")

    return expanded

# ═══════════════════════════════════════════════════════════════════════════
# Main execution
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("Massive NLU v6 Generation — Target 65k+ rows")
    print("=" * 80)

    # Stage 1: Context multiplication
    print("\n[1/3] Applying contextual multipliers...")
    expanded = multiply_contexts(v5_rows)

    # Stage 2: Regional + arabizi (from expand_datasets_v6.py)
    print("\n[2/3] Applying regional/arabizi expansion...")
    from expand_datasets_v6 import (
        apply_regional_variant, transliterate_to_arabizi, inject_typos
    )

    stage2_base = len(expanded)
    regions = ['west', 'tlemcen', 'east', 'south']

    # Regional (20% of current)
    for row in random.sample(expanded[:stage2_base], min(len(expanded)//5, 8000)):
        region = random.choice(regions)
        regional = apply_regional_variant(row['text'], region)
        if regional != row['text']:
            expanded.append({
                'text': regional,
                'intent': row['intent'],
                'profession': row['profession'],
                'source': f'synth_v6_regional_{region}'
            })

    # Arabizi (25% of Arabic rows)
    ar_rows = [r for r in expanded[:stage2_base] if is_arabic_script(r['text'])]
    for row in random.sample(ar_rows, min(len(ar_rows)//4, 10000)):
        arabizi = transliterate_to_arabizi(row['text'])
        expanded.append({
            'text': arabizi,
            'intent': row['intent'],
            'profession': row['profession'],
            'source': 'synth_v6_arabizi'
        })

    # Typos (8% of all)
    for row in random.sample(expanded[:stage2_base], min(len(expanded)//12, 5000)):
        typo = inject_typos(row['text'], prob=0.08)
        if typo != row['text']:
            expanded.append({
                'text': typo,
                'intent': row['intent'],
                'profession': row['profession'],
                'source': 'synth_v6_typos'
            })

    print(f"After regional/arabizi/typos: {len(expanded)} rows")

    # Stage 3: Dedup + quality filter
    print("\n[3/3] Deduplication and quality filtering...")

    # Exact dedup
    seen = set()
    deduped = []
    for row in expanded:
        key = (row['text'].strip().lower(), row['intent'], row['profession'])
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    print(f"  After exact dedup: {len(deduped)} rows")

    # Length filter
    filtered = [r for r in deduped if 5 <= len(r['text']) <= 250]
    print(f"  After length filter: {len(filtered)} rows")

    # Save
    out_path = OUT / 'synth_v6_massive.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'intent', 'profession', 'source'])
        writer.writeheader()
        writer.writerows(filtered)

    print(f"\nSaved to {out_path}")
    print(f"Final count: {len(filtered)} rows ({len(filtered)/len(v5_rows):.1f}x base)")
    print(f"\nSource breakdown:")
    for src, count in Counter(r['source'] for r in filtered).most_common():
        print(f"  {src}: {count}")

if __name__ == '__main__':
    main()
