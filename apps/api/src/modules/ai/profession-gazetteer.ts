// apps/api/src/modules/ai/profession-gazetteer.ts
//
// Gazetier des MÉTIERS — complément du NLU SANS entraînement, même design que
// le gazetier wilaya (gazetteer.ts) : détecte une mention de profession dans un
// texte libre (darija arabe/latin, fr) et retourne la clé canonique.
//
// POURQUOI (bug terrain 20/08) : « خاصني بلومبي » Soni transcrit parfaitement
// par le CTC (0,6236 WER servi), mais le NLU donnait 0 % : « بلومبي »
// n'existe pas dans le lexique appris (le dataset n'a que « بلومبيي » avec un
// ي parasite) → confiance sous le gate 0.35 → FALLBACK. La vraie faiblesse
// est le LEXIQUE NLU, pas l'audio. Le gazetier rehausse la profession à
// partir des noms/symptômes réels des utilisateurs.
//
// Normalisation partagée avec gazetteer.ts (mêmes règles : accents, harakat,
// hamza→alif, ى→ي, ة→ه) — mention = mot entier, alias le plus long d'abord,
// préfixe arabe collé optionnel (فالبلومبي، بالسباك…).
//
// Hiérarchie d'usage (intent-extractor.service.ts) :
//   NLU conf ≥ gate 0.35  → NLU (le plus riche)
//   NLU conf < 0.35       → gazetier (rehausse si hit, sinon FALLBACK)
//   NLU absent            → gazetier seul (dégradation, jamais de panne)

// [clé canonique (alignée VALID_PROFESSIONS + labels NLU + seeder), aliases]
// Les aliases arabes commençant par "ال" génèrent la variante sans article.
// RÈGLE : n'inclure QUE des mots réellement prononcés (pas d'invention) — le
// corpus lm_domaine.txt (ml/kaggle) et le lexique NLU (ml/generate_dataset.py)
// sont la source ; compléter ici signale un trou à corriger DANS LE DATASET
// (P6 flywheel), pas seulement ici.
const P: Array<[string, string[]]> = [
  ['plumber', ['سباك', 'بلومبي', 'بلومبيي', 'مول السباكة', 'plombier', 'plombi',
               'sebbak', 'روبنيه', 'روبينية', 'لافابو', 'بالوعة', 'طوفان الما',
               'يسيل', 'فويت', 'y9atter', 'msdouda']],
  ['electrician', ['كهربائي', 'تريسيان', 'مول الضو', 'الضوء', 'فينوز',
                   'فيوز', 'كهرباء', 'برايز', 'electricien', 'electrician',
                   'مول النور', 'العداد']],
  ['ac_repair', ['كليماتيزور', 'مكيف', 'تقني كليماتيزور', 'مول الكليمة',
                 'التكييف', 'clim', 'reparateur clim', 'فريون']],
  ['mason', ['بناي', 'بنّاء', 'ماصون', 'مول البناء', 'bennay', 'maçon',
             'maçon', 'حائط', 'جدار', 'طوب']],
  ['painter', ['صباغ', 'دهان', 'مول الصباغة', 'sabbagh', 'peintre',
               'الدهان', 'الصباغة', 'بينتور']],
  ['carpenter', ['نجار', 'منوزيي', 'مول الخشب', 'najjar', 'menuisier',
                 'لخشب']],
  ['cleaner', ['عاملة تنظيف', 'فام دو ميناج', 'ماشينة تنظيف', 'ménage',
               'menage', 'مول الميناج', 'تنظيف']],
  ['appliance_repair', ['مصلح الأجهزة', 'مصلح تلاجات', 'ماشينة خاسرة',
                        'تصليح الماشينة', 'غسالة خاسرة', 'مصلح ماشينة',
                        'reparateur machine', 'مكواة مكسورة', 'الفرجيدار',
                        'تلاجة خاسرة', 'الفرن خاسر', 'التلاجة', 'مكواة خاسرة']],
  ['mover', ['ديمناجمون', 'ديمناجور', 'ناقل العفش', 'نقل الاثاث',
             'demenageur', 'نقل العفش']],
  ['mechanic', ['ميكانيكي', 'ميكانيسيان', 'مول الميكانيك', 'ميكانيك',
                'mécanicien', 'mecanicien', 'دور مقطوع', 'البانشيه',
                'ماطور خاسر']],
  ['plasterer', ['جباص', 'بلاكيست', 'مول الجبس', 'مول البلاكو', 'بلاكو',
                 'الجبس']],
  ['welder', ['سودور', 'لحّام', 'مول السودير', 'soudeur', 'الحديد', 'لحام']],
  ['barber', ['حلاق', 'كوافور', 'مول الحلاقة', 'الحلاقة', 'حلاقة']],
  ['tailor', ['خياط', 'خياطة', 'مول الخياطة', 'tailleur', 'القفطان',
              'خياطة']],
  ['caterer', ['طباخة', 'تريتور', 'مول الطياب', 'طباخ', 'traiteur',
               'cuisinière de mariage', 'طبخ']],
];

function norm(s: string): string {
  return s
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '') // accents latins decomposes (e\u0301 -> e)
    .replace(/[\u064b-\u0655\u0670\u0640]/g, '') // harakat, hamza combines, tatweel
    .replace(/[\u0623\u0625\u0622]/g, '\u0627') // hamza-alif -> alif
    .replace(/\u0649/g, '\u064a') // alif maqsura -> ya
    .replace(/\u0629/g, '\u0647') // ta marbuta -> ha
    .replace(/[^a-z0-9\u0600-\u06ff]+/g, ' ')
    .trim();
}

// Index construit une fois : alias normalisé → clé canonique.
const ALIAS_MAP = new Map<string, string>();
for (const [key, aliases] of P) {
  for (const a of aliases) {
    const n = norm(a);
    for (const v of n.startsWith('ال') ? [n, n.slice(2)] : [n]) {
      if (!ALIAS_MAP.has(v)) ALIAS_MAP.set(v, key); // variante sans article
    }
  }
}

// Même structure que le gazetier wilaya : bornes espace + préfixe arabe collé
// optionnel. Les alias normalisés ne contiennent que lettres/chiffres/espaces
// — pas d'échappement regex nécessaire.
const PATTERN = new RegExp(
  ' (?:وال|فال|بال|لل|ال|و|ف|ب|ل)?(' +
    [...ALIAS_MAP.keys()].sort((a, b) => b.length - a.length).join('|') +
  ') ',
);

/** Première profession mentionnée dans [text] — null si aucune. */
export function findProfession(text: string): string | null {
  const m = PATTERN.exec(` ${norm(text)} `);
  return m ? ALIAS_MAP.get(m[1])! : null;
}