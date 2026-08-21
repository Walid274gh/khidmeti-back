#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
# ml/merge_and_augment_v6.py — Final merge + augmentation to 65k+
#
# Combines:
#   - synth_v5.csv (6.5k baseline)
#   - synth_v6_massive.csv (20k with contexts)
#   - synth_v6_lexicon.csv (17k exhaustive)
#   - hand_v4.csv (47 real examples)
#
# Then applies full augmentation battery:
#   - Regional variants (4 regions)
#   - Arabizi transliteration
#   - Typo injection
#   - Synonym substitution
#   - Word order permutation (Darija flexible syntax)
#
# Target: 65k+ final training rows
# ══════════════════════════════════════════════════════════════════════════════
import csv
import json
import random
import re
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

random.seed(20260810)

OUT = Path(__file__).resolve().parent / 'dataset'
LABELS = json.loads((OUT / 'labels.json').read_text())

# Load all sources
print("Loading all data sources...")
sources = {
    'v5_base': list(csv.DictReader(open(OUT / 'synth_v5.csv'))),
    'v6_massive': list(csv.DictReader(open(OUT / 'synth_v6_massive.csv'))),
    'v6_lexicon': list(csv.DictReader(open(OUT / 'synth_v6_lexicon.csv'))),
    'hand_v4': list(csv.DictReader(open(OUT / 'hand_v4.csv'))),
}

for name, rows in sources.items():
    print(f"  {name}: {len(rows)} rows")

# Merge with source tracking
all_rows = []
for source_name, rows in sources.items():
    for row in rows:
        all_rows.append({
            'text': row['text'],
            'intent': row['intent'],
            'profession': row['profession'],
            'source': row.get('source', source_name),
        })

print(f"Total merged: {len(all_rows)} rows")

# ═══════════════════════════════════════════════════════════════════════════
# Augmentation functions
# ═══════════════════════════════════════════════════════════════════════════

REGIONAL_SWAPS = {
    'west': {
        'نحب': 'نبغي', 'بزاف': 'برشا', 'مليح': 'مزيان', 'توا': 'دابا',
        'كيفاش': 'كيفاه', 'وين': 'فين', 'هذا': 'هادا', 'درك': 'دابا',
        'واش': 'واخا', 'شوية': 'شوي', 'الدار': 'الديار',
    },
    'tlemcen': {
        'قهوة': 'اهوة', 'قلب': 'الب', 'قال': 'ال', 'قدام': 'ادام',
        'قريب': 'اريب', 'قديم': 'اديم', 'قاع': 'اع',
    },
    'east': {
        'بزاف': 'ياسر', 'كيفاش': 'كيفاه', 'درك': 'دورك', 'حلو': 'زين',
        'مليح': 'برشا', 'توا': 'دورك', 'وين': 'اينا',
    },
    'south': {
        'مليح': 'زين', 'بزاف': 'دروك', 'روح': 'سير', 'واش': 'وش',
        'شوية': 'شوي', 'كيفاش': 'كيف',
    },
}

AR_TO_ARABIZI = {
    'ا': 'a', 'أ': 'a', 'إ': 'a', 'آ': 'a', 'ب': 'b', 'ت': 't',
    'ث': 'th', 'ج': 'j', 'ح': '7', 'خ': 'kh', 'د': 'd', 'ذ': 'dh',
    'ر': 'r', 'ز': 'z', 'س': 's', 'ش': 'ch', 'ص': 's', 'ض': 'd',
    'ط': 't', 'ظ': 'z', 'ع': '3', 'غ': 'gh', 'ف': 'f', 'ق': '9',
    'ك': 'k', 'ل': 'l', 'م': 'm', 'ن': 'n', 'ه': 'h', 'و': 'w',
    'ي': 'y', 'ة': 'a', 'ى': 'a', ' ': ' ',
}

SYNONYM_SWAPS_AR = {
    'نحتاج': ['نحوس', 'نبغي', 'خصني', 'لازم'],
    'مليح': ['مزيان', 'برشا', 'بصح', 'شاطر'],
    'بزاف': ['برشا', 'ياسر', 'واجد', 'كثير'],
    'الدار': ['البيت', 'المنزل', 'الفيلا'],
    'مشكل': ['مشكلة', 'عطل', 'خلل', 'بان'],
    'يصلح': ['يسالح', 'يرانجي', 'يخدم'],
}

SYNONYM_SWAPS_LAT = {
    'n7taj': ['n7awes', 'bghit', 'khesni', 'lazem'],
    'mli7': ['mzyan', 'barcha', 'chater', 'nchit'],
    'bezzaf': ['barcha', 'yasser', 'wajed', 'kthir'],
    'dar': ['bit', 'villa', 'appartement'],
    'mochkil': ['mochkla', '3otel', 'pan', 'khala'],
}

def is_arabic_script(text: str) -> bool:
    return any('؀' <= c <= 'ۿ' for c in text)

def apply_regional_variant(text: str, region: str) -> str:
    if region not in REGIONAL_SWAPS:
        return text
    swaps = REGIONAL_SWAPS[region]
    for old, new in swaps.items():
        text = text.replace(old, new)
    return text

def transliterate_to_arabizi(text: str) -> str:
    result = []
    for char in text:
        result.append(AR_TO_ARABIZI.get(char, char))
    base = ''.join(result)

    # Qaf variants
    if '9' in base:
        variant = random.choice([
            base,
            base.replace('9', 'q'),
            base.replace('9', 'g'),
            base.replace('9', '2'),
        ])
        return variant
    return base

def inject_typos(text: str, prob=0.06) -> str:
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < prob:
            r = random.random()
            if r < 0.35 and i < len(chars) - 1:  # swap
                chars[i], chars[i+1] = chars[i+1], chars[i]
            elif r < 0.65 and i < len(chars) - 1:  # repeat
                chars[i] = chars[i] * 2
            elif r < 0.90:  # delete
                chars[i] = ''
    return ''.join(chars)

def apply_synonym_swap(text: str) -> str:
    swaps = SYNONYM_SWAPS_AR if is_arabic_script(text) else SYNONYM_SWAPS_LAT
    words = text.split()
    changed = False
    for i, word in enumerate(words):
        if word in swaps and random.random() < 0.3:
            words[i] = random.choice(swaps[word])
            changed = True
    return ' '.join(words) if changed else text

def permute_word_order(text: str, prob=0.05) -> str:
    """Darija allows flexible word order for some constructions."""
    if random.random() > prob:
        return text
    words = text.split()
    if len(words) < 3:
        return text

    # Simple permutation: swap first two or last two words
    if random.random() < 0.5 and len(words) >= 2:
        words[0], words[1] = words[1], words[0]
    elif len(words) >= 3:
        words[-2], words[-1] = words[-1], words[-2]

    return ' '.join(words)

# ═══════════════════════════════════════════════════════════════════════════
# Augmentation pipeline
# ═══════════════════════════════════════════════════════════════════════════

print("\nApplying augmentation pipeline...")

augmented = list(all_rows)  # Start with base
base_size = len(augmented)

# Pass 1: Regional variants (target 4x coverage of 30% of rows)
print("  [1/5] Regional variants...")
sample_regional = random.sample(all_rows, min(len(all_rows), 15000))
for row in sample_regional:
    for region in ['west', 'tlemcen', 'east', 'south']:
        regional_text = apply_regional_variant(row['text'], region)
        if regional_text != row['text']:
            augmented.append({
                'text': regional_text,
                'intent': row['intent'],
                'profession': row['profession'],
                'source': f'aug_regional_{region}',
            })
print(f"     Added {len(augmented) - base_size} variants")

# Pass 2: Arabizi (all Arabic rows)
print("  [2/5] Arabizi transliteration...")
ar_count = len(augmented)
ar_rows = [r for r in all_rows if is_arabic_script(r['text'])]
for row in ar_rows:
    arabizi = transliterate_to_arabizi(row['text'])
    augmented.append({
        'text': arabizi,
        'intent': row['intent'],
        'profession': row['profession'],
        'source': 'aug_arabizi',
    })
print(f"     Added {len(augmented) - ar_count} variants")

# Pass 3: Synonym swaps (20% of rows)
print("  [3/5] Synonym substitution...")
syn_count = len(augmented)
for row in random.sample(all_rows, min(len(all_rows)//5, 10000)):
    syn_text = apply_synonym_swap(row['text'])
    if syn_text != row['text']:
        augmented.append({
            'text': syn_text,
            'intent': row['intent'],
            'profession': row['profession'],
            'source': 'aug_synonym',
        })
print(f"     Added {len(augmented) - syn_count} variants")

# Pass 4: Typos (15% of current total)
print("  [4/5] Typo injection...")
typo_count = len(augmented)
for row in random.sample(augmented[:base_size], min(len(augmented)//7, 8000)):
    typo_text = inject_typos(row['text'], prob=0.07)
    if typo_text != row['text']:
        augmented.append({
            'text': typo_text,
            'intent': row['intent'],
            'profession': row['profession'],
            'source': 'aug_typo',
        })
print(f"     Added {len(augmented) - typo_count} variants")

# Pass 5: Word order permutation (5% of rows)
print("  [5/5] Word order permutation...")
perm_count = len(augmented)
for row in random.sample(augmented[:base_size], min(len(augmented)//20, 3000)):
    perm_text = permute_word_order(row['text'], prob=1.0)
    if perm_text != row['text']:
        augmented.append({
            'text': perm_text,
            'intent': row['intent'],
            'profession': row['profession'],
            'source': 'aug_permute',
        })
print(f"     Added {len(augmented) - perm_count} variants")

print(f"\nTotal after augmentation: {len(augmented)} rows")

# ═══════════════════════════════════════════════════════════════════════════
# Deduplication and quality filter
# ═══════════════════════════════════════════════════════════════════════════

print("\nDeduplication and quality filtering...")

# Exact dedup
seen = set()
deduped = []
for row in augmented:
    norm_text = row['text'].strip().lower()
    norm_text = re.sub(r'\s+', ' ', norm_text)
    key = (norm_text, row['intent'], row['profession'])
    if key not in seen:
        seen.add(key)
        deduped.append(row)

print(f"  After exact dedup: {len(deduped)} rows")

# Length filter
filtered = [r for r in deduped if 3 <= len(r['text']) <= 300]
print(f"  After length filter: {len(filtered)} rows")

# Character set filter (Arabic/Latin/digits/common punctuation only)
valid_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                  + 'ابتثجحخدذرزسشصضطظعغفقكلمنهويةىأإآئءؤ'
                  + ' .,?!:;-_()/\'\"')
charset_filtered = []
for r in filtered:
    if sum(1 for c in r['text'] if c in valid_chars) / max(len(r['text']), 1) > 0.7:
        charset_filtered.append(r)

print(f"  After charset filter: {len(charset_filtered)} rows")

# Save final dataset
out_path = OUT / 'synth_v6_final.csv'
with open(out_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['text', 'intent', 'profession', 'source'])
    writer.writeheader()
    writer.writerows(charset_filtered)

print(f"\nSaved final dataset to {out_path}")
print(f"Final size: {len(charset_filtered)} rows")
print(f"  vs v5 baseline 6581 = {len(charset_filtered)/6581:.1f}x")

print(f"\nIntent distribution:")
for intent, count in Counter(r['intent'] for r in charset_filtered).most_common():
    print(f"  {intent}: {count}")

print(f"\nProfession distribution:")
for prof, count in Counter(r['profession'] for r in charset_filtered).most_common():
    print(f"  {prof}: {count}")

print(f"\nSource distribution (top 10):")
for src, count in Counter(r['source'] for r in charset_filtered).most_common(10):
    print(f"  {src}: {count}")

# Create stratified splits
print("\nCreating stratified train/val/test splits...")
by_prof = defaultdict(list)
for i, row in enumerate(charset_filtered):
    by_prof[row['profession']].append(i)

train_idx, val_idx, test_idx = [], [], []
for prof, indices in by_prof.items():
    random.shuffle(indices)
    n = len(indices)
    n_test = max(1, int(n * 0.10))
    n_val = max(1, int(n * 0.05))

    test_idx.extend(indices[:n_test])
    val_idx.extend(indices[n_test:n_test+n_val])
    train_idx.extend(indices[n_test+n_val:])

splits_path = OUT / 'splits_v6_final.json'
json.dump({
    'train': train_idx,
    'val': val_idx,
    'test': test_idx,
    'seed': 20260810,
    'train_size': len(train_idx),
    'val_size': len(val_idx),
    'test_size': len(test_idx),
}, open(splits_path, 'w'), indent=2)

print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
print(f"  Splits saved to {splits_path}")

print("\n" + "=" * 80)
print("DATASET V6 COMPLETE")
print("=" * 80)
print(f"Final training dataset: {out_path}")
print(f"Total rows: {len(charset_filtered)}")
print(f"Ready for NLU training with raised gate: prof_acc >= 0.90")
