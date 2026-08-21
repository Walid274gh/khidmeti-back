#!/usr/bin/env python3
"""Validation pré-push v11 : labels, overlaps, charset (leçon homoglyphes), dups."""
import csv, json, re
from pathlib import Path

ds = Path(__file__).resolve().parent / 'dataset'
L = json.load(open(ds / 'labels.json'))
print('labels:', len(L['professions']), 'profs,', len(L['intents']), 'intents')
assert len(L['professions']) == 16 and 'none' in L['professions'], L['professions']


def load(f):
    return list(csv.DictReader(open(ds / f, encoding='utf-8')))


ev, h4, s5 = load('eval_heldout.csv'), load('hand_v4.csv'), load('synth_v5.csv')
h123 = load('hand_v1.csv') + load('hand_v2.csv') + load('hand_v3.csv')
print(f'eval={len(ev)} hand_v4={len(h4)} synth_v5={len(s5)} hand_v1-3={len(h123)}')

for name, rowset in [('hand_v4', h4), ('synth_v5', s5), ('eval', ev), ('hand_v1-3', h123)]:
    for r in rowset:
        assert r['intent'] in L['intents'], (name, r)
        assert r['profession'] in L['professions'], (name, r)

evt = {r['text'] for r in ev}
for name, rowset in [('hand_v4', h4), ('synth_v5', s5), ('hand_v1-3', h123)]:
    ov = {r['text'] for r in rowset} & evt
    print(f'{name} x eval overlap = {len(ov)}', list(ov)[:3])
    assert not ov, ov
h4t = {r['text'] for r in h4}
ov = (h4t & {r['text'] for r in s5}) | (h4t & {r['text'] for r in h123})
print('hand_v4 x (synth_v5+hand_v1-3) =', len(ov))
assert not ov, ov

# charset : ASCII + bloc arabe + suppléments + accents fr — attrape les homoglyphes
ok = re.compile(u'^[ -~؀-ۿݐ-ݿ’…àçèéêîôûù«»\\s]+$')
bad = []
for n, rs in [('hand_v4', h4), ('eval', ev)]:
    for r in rs:
        if not ok.match(r['text']):
            oddc = [hex(ord(c)) for c in r['text'] if not ok.match(c)]
            bad.append((n, r['text'], oddc))
for b in bad[:10]:
    print('CHARSET?', b)
assert not bad, f'{len(bad)} suspect rows'

texts = [r['text'] for r in ev]
assert len(texts) == len(set(texts)), 'dup in eval'

joined = ' '.join(r['text'] for r in s5)
for m in ('magtou3', 'ma9tou3', 'daw', 'y9atter', 'mchit'):
    print(m, 'in synth_v5:', m in joined)
print('ALL CHECKS PASS')
