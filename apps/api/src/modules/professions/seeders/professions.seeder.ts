// ══════════════════════════════════════════════════════════════════════════════
// INITIAL_PROFESSIONS — Seed data
//
// ⚠️  CONTRAINTE CRITIQUE : les `key` ici DOIVENT correspondre exactement aux
//     valeurs dans VALID_PROFESSIONS (intent-extractor.service.ts).
//     Toute divergence provoquera des silences silencieux dans le matching IA.
//
// Catégories :
//   water     → eau / plomberie
//   energy    → électricité / climatisation
//   building  → bâtiment / structure
//   service   → services à domicile
//   transport → mobilité
// ══════════════════════════════════════════════════════════════════════════════

import { professionHero } from './profession-media';

export interface ProfessionSeedItem {
  key: string;
  iconName: string;
  categoryKey: string;
  isActive: boolean;
  sortOrder: number;
  labels: { fr: string; ar: string; en: string };
  categoryLabels: { fr: string; ar: string; en: string };
  /** Photo du métier (profession-media.ts) — remplacée par Cloudinary dès que l'admin upload. */
  imageUrl?: string;
}

const CATEGORY_LABELS: Record<string, { fr: string; ar: string; en: string }> = {
  water:     { fr: 'Eau & Plomberie',      ar: 'المياه والسباكة',       en: 'Water & Plumbing'    },
  energy:    { fr: 'Énergie & Froid',      ar: 'الطاقة والتبريد',       en: 'Energy & Cooling'    },
  building:  { fr: 'Bâtiment & Rénovation',ar: 'البناء والتجديد',       en: 'Building & Renovation'},
  service:   { fr: 'Services à domicile',  ar: 'الخدمات المنزلية',      en: 'Home Services'       },
  transport: { fr: 'Transport & Mobilité', ar: 'النقل والتنقل',         en: 'Transport & Mobility' },
};

const PROFESSIONS_WITHOUT_IMAGE: Omit<ProfessionSeedItem, 'imageUrl'>[] = [
  // ── Eau & Plomberie ────────────────────────────────────────────────────────
  {
    key:           'plumber',
    iconName:      'water_drop_outlined',
    categoryKey:   'water',
    isActive:      true,
    sortOrder:     10,
    labels:        { fr: 'Plombier',     ar: 'سباك',          en: 'Plumber'           },
    categoryLabels: CATEGORY_LABELS.water,
  },

  // ── Énergie & Froid ───────────────────────────────────────────────────────
  {
    key:           'electrician',
    iconName:      'bolt_outlined',
    categoryKey:   'energy',
    isActive:      true,
    sortOrder:     20,
    labels:        { fr: 'Électricien',  ar: 'كهربائي',       en: 'Electrician'       },
    categoryLabels: CATEGORY_LABELS.energy,
  },
  {
    key:           'ac_repair',
    iconName:      'air_outlined',
    categoryKey:   'energy',
    isActive:      true,
    sortOrder:     21,
    labels:        { fr: 'Climatisation', ar: 'تكييف الهواء', en: 'AC Repair'          },
    categoryLabels: CATEGORY_LABELS.energy,
  },

  // ── Bâtiment & Rénovation ─────────────────────────────────────────────────
  {
    key:           'mason',
    iconName:      'foundation_outlined',
    categoryKey:   'building',
    isActive:      true,
    sortOrder:     30,
    labels:        { fr: 'Maçon',        ar: 'بنّاء',          en: 'Mason'              },
    categoryLabels: CATEGORY_LABELS.building,
  },
  {
    key:           'painter',
    iconName:      'format_paint_outlined',
    categoryKey:   'building',
    isActive:      true,
    sortOrder:     31,
    labels:        { fr: 'Peintre',      ar: 'دهان',           en: 'Painter'            },
    categoryLabels: CATEGORY_LABELS.building,
  },
  {
    key:           'carpenter',
    iconName:      'carpenter_outlined',
    categoryKey:   'building',
    isActive:      true,
    sortOrder:     32,
    labels:        { fr: 'Menuisier',    ar: 'نجار',           en: 'Carpenter'          },
    categoryLabels: CATEGORY_LABELS.building,
  },
  {
    key:           'plasterer',
    iconName:      'layers_outlined',
    categoryKey:   'building',
    isActive:      true,
    sortOrder:     33,
    labels:        { fr: 'Plaquiste (BA13)', ar: 'بلاكيست (BA13)', en: 'Drywall installer' },
    categoryLabels: CATEGORY_LABELS.building,
  },
  {
    key:           'welder',
    iconName:      'local_fire_department_outlined',
    categoryKey:   'building',
    isActive:      true,
    sortOrder:     34,
    labels:        { fr: 'Soudeur',      ar: 'لحّام (سودور)',   en: 'Welder'             },
    categoryLabels: CATEGORY_LABELS.building,
  },

  // ── Services à domicile ───────────────────────────────────────────────────
  {
    key:           'cleaner',
    iconName:      'cleaning_services_outlined',
    categoryKey:   'service',
    isActive:      true,
    sortOrder:     40,
    labels:        { fr: 'Agent de nettoyage', ar: 'عامل تنظيف', en: 'Cleaner'         },
    categoryLabels: CATEGORY_LABELS.service,
  },
  {
    key:           'appliance_repair',
    iconName:      'home_repair_service_outlined',
    categoryKey:   'service',
    isActive:      true,
    sortOrder:     41,
    labels:        { fr: 'Réparation électroménager', ar: 'إصلاح الأجهزة المنزلية', en: 'Appliance Repair' },
    categoryLabels: CATEGORY_LABELS.service,
  },
  {
    key:           'mover',
    iconName:      'local_shipping_outlined',
    categoryKey:   'service',
    isActive:      true,
    sortOrder:     43,
    labels:        { fr: 'Déménageur',   ar: 'ناقل عفش',        en: 'Mover'              },
    categoryLabels: CATEGORY_LABELS.service,
  },
  {
    key:           'barber',
    iconName:      'content_cut_outlined',
    categoryKey:   'service',
    isActive:      true,
    sortOrder:     44,
    labels:        { fr: 'Coiffeur',     ar: 'حلاق',            en: 'Barber'             },
    categoryLabels: CATEGORY_LABELS.service,
  },
  {
    key:           'tailor',
    iconName:      'checkroom_outlined',
    categoryKey:   'service',
    isActive:      true,
    sortOrder:     45,
    labels:        { fr: 'Couturier',    ar: 'خيّاط',           en: 'Tailor'             },
    categoryLabels: CATEGORY_LABELS.service,
  },
  {
    key:           'caterer',
    iconName:      'restaurant_outlined',
    categoryKey:   'service',
    isActive:      true,
    sortOrder:     46,
    labels:        { fr: 'Traiteur',     ar: 'طبّاخة (تريتور)', en: 'Caterer'            },
    categoryLabels: CATEGORY_LABELS.service,
  },

  // ── Transport & Mobilité ──────────────────────────────────────────────────
  {
    key:           'mechanic',
    iconName:      'car_repair_outlined',
    categoryKey:   'transport',
    isActive:      true,
    sortOrder:     50,
    labels:        { fr: 'Mécanicien',   ar: 'ميكانيكي',        en: 'Mechanic'           },
    categoryLabels: CATEGORY_LABELS.transport,
  },
];

/** Image dérivée du métier — un seul endroit décide des photos (profession-media.ts). */
export const INITIAL_PROFESSIONS: ProfessionSeedItem[] = PROFESSIONS_WITHOUT_IMAGE.map(
  (p) => ({ ...p, imageUrl: professionHero(p.key) }),
);
