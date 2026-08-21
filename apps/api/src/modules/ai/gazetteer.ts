// apps/api/src/modules/ai/gazetteer.ts
//
// Gazetier des 58 wilayas — extraction de lieu SANS modèle (P3 pipeline darija).
// Détecte une mention de wilaya dans un texte libre (darija arabe/latin, fr)
// et retourne { code, name } — le code numérique est celui utilisé partout
// (cellId, wilayaCodes de /search/workers, WilayaManager Flutter).
//
// Normalisation partagée texte/alias : minuscules, accents latins retirés
// (é→e), harakat/tatweel retirés, hamza-alif unifiés (أإآ→ا), ى→ي, ة→ه,
// tout séparateur → espace. Le match est borné par espaces (mots entiers),
// alias les plus longs testés d'abord ("برج بوعريريج" avant tout sous-mot).
//
// ponytail: chefs-lieux + variantes courantes uniquement — pas les 1541
// communes; en ajouter au tableau W si le flywheel P6 montre des manques.

export interface WilayaHit {
  code: number;
  name: string;
}

// [code, nom latin (aligné WilayaManager Flutter), aliases]
// Les alias arabes commençant par "ال" génèrent automatiquement la variante
// sans article (البليدة → بليدة) — inutile de doubler à la main.
const W: Array<[number, string, string[]]> = [
  [1,  'Adrar',               ['adrar', 'أدرار']],
  [2,  'Chlef',               ['chlef', 'الشلف', 'chleff']],
  [3,  'Laghouat',            ['laghouat', 'الأغواط', 'لغواط']],
  [4,  'Oum El Bouaghi',      ['oum el bouaghi', 'أم البواقي']],
  [5,  'Batna',               ['batna', 'باتنة']],
  [6,  'Béjaïa',              ['bejaia', 'bgayet', 'بجاية', 'بگايت']],
  [7,  'Biskra',              ['biskra', 'بسكرة']],
  [8,  'Béchar',              ['bechar', 'بشار']],
  [9,  'Blida',               ['blida', 'البليدة']],
  [10, 'Bouira',              ['bouira', 'البويرة', 'tubiret']],
  [11, 'Tamanrasset',         ['tamanrasset', 'tamanghasset', 'تمنراست']],
  [12, 'Tébessa',             ['tebessa', 'تبسة']],
  [13, 'Tlemcen',             ['tlemcen', 'تلمسان']],
  [14, 'Tiaret',              ['tiaret', 'تيارت']],
  [15, 'Tizi Ouzou',          ['tizi ouzou', 'تيزي وزو']],
  [16, 'Alger',               ['alger', 'algiers', 'العاصمة', 'الجزائر العاصمة', 'dzayer']],
  [17, 'Djelfa',              ['djelfa', 'الجلفة']],
  [18, 'Jijel',               ['jijel', 'جيجل']],
  [19, 'Sétif',               ['setif', 'stif', 'سطيف']],
  [20, 'Saïda',               ['saida', 'سعيدة']],
  [21, 'Skikda',              ['skikda', 'سكيكدة']],
  [22, 'Sidi Bel Abbès',      ['sidi bel abbes', 'سيدي بلعباس', 'bel abbes']],
  [23, 'Annaba',              ['annaba', 'عنابة']],
  [24, 'Guelma',              ['guelma', 'قالمة']],
  [25, 'Constantine',         ['constantine', 'قسنطينة', 'ksantina', 'cirta']],
  [26, 'Médéa',               ['medea', 'المدية']],
  [27, 'Mostaganem',          ['mostaganem', 'مستغانم']],
  [28, "M'Sila",              ['msila', 'المسيلة']],
  [29, 'Mascara',             ['mascara', 'معسكر', 'mouaskar']],
  [30, 'Ouargla',             ['ouargla', 'ورقلة', 'wargla']],
  [31, 'Oran',                ['oran', 'وهران', 'wahran', 'wehran']],
  [32, 'El Bayadh',           ['el bayadh', 'البيض']],
  [33, 'Illizi',              ['illizi', 'إليزي']],
  [34, 'Bordj Bou Arréridj',  ['bordj bou arreridj', 'برج بوعريريج', 'bba']],
  [35, 'Boumerdès',           ['boumerdes', 'بومرداس']],
  [36, 'El Tarf',             ['el tarf', 'الطارف']],
  [37, 'Tindouf',             ['tindouf', 'تندوف']],
  [38, 'Tissemsilt',          ['tissemsilt', 'تيسمسيلت']],
  [39, 'El Oued',             ['el oued', 'الوادي', 'oued souf', 'souf']],
  [40, 'Khenchela',           ['khenchela', 'خنشلة']],
  [41, 'Souk Ahras',          ['souk ahras', 'سوق أهراس']],
  [42, 'Tipaza',              ['tipaza', 'tipasa', 'تيبازة']],
  [43, 'Mila',                ['mila', 'ميلة']],
  [44, 'Aïn Defla',           ['ain defla', 'عين الدفلى']],
  [45, 'Naâma',               ['naama', 'النعامة']],
  [46, 'Aïn Témouchent',      ['ain temouchent', 'عين تموشنت']],
  [47, 'Ghardaïa',            ['ghardaia', 'غرداية']],
  [48, 'Relizane',            ['relizane', 'غليزان', 'ghilizane']],
  [49, 'Timimoun',            ['timimoun', 'تيميمون']],
  [50, 'Bordj Badji Mokhtar', ['bordj badji mokhtar', 'برج باجي مختار']],
  [51, 'Ouled Djellal',       ['ouled djellal', 'أولاد جلال']],
  [52, 'Béni Abbès',          ['beni abbes', 'بني عباس']],
  [53, 'In Salah',            ['in salah', 'ain salah', 'عين صالح']],
  [54, 'In Guezzam',          ['in guezzam', 'عين قزام']],
  [55, 'Touggourt',           ['touggourt', 'tougourt', 'تقرت']],
  [56, 'Djanet',              ['djanet', 'جانت']],
  [57, "El M'Ghair",          ['el mghair', 'المغير']],
  [58, 'El Meniaa',           ['el meniaa', 'el menia', 'المنيعة']],
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

// Index construit une fois : alias normalisé → hit. Le match passe par UN seul
// regex : frontière-espace + préfixe arabe collé optionnel (فالبليدة، بسطيف،
// وهران…) + alternation des alias triés du plus long au plus court (le
// multi-mots gagne). Les alias normalisés ne contiennent que lettres/chiffres/
// espaces — pas d'échappement regex nécessaire.
const ALIAS_MAP = new Map<string, WilayaHit>();
for (const [code, name, aliases] of W) {
  for (const a of aliases) {
    const n = norm(a);
    for (const v of n.startsWith('ال') ? [n, n.slice(2)] : [n]) {
      if (!ALIAS_MAP.has(v)) ALIAS_MAP.set(v, { code, name }); // variante sans article
    }
  }
}

const PATTERN = new RegExp(
  ' (?:وال|فال|بال|لل|ال|و|ف|ب|ل)?(' +
    [...ALIAS_MAP.keys()].sort((a, b) => b.length - a.length).join('|') +
  ') ',
);

/** Première wilaya mentionnée dans [text] (alias le plus long prioritaire). */
export function findWilaya(text: string): WilayaHit | null {
  const m = PATTERN.exec(` ${norm(text)} `);
  return m ? { ...ALIAS_MAP.get(m[1])! } : null;
}
