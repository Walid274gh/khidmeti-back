// ══════════════════════════════════════════════════════════════════════════════
// KHIDMETI — Script de seed : travailleurs de test sur les 58 wilayas
//
// USAGE via Makefile (recommandé) :
//   make scripts-seed-workers              ← seed / re-seed (upsert)
//   make scripts-seed-workers ARGS=--clear ← efface tout puis re-seed
//
// PROPRIÉTÉS :
//   • ~750 workers répartis sur les 58 wilayas (villes majeures densifiées),
//     en 2 anneaux : 1/3 près du centre (±9 km), 2/3 sur toute la wilaya
//     (±39 km) — un client loin du chef-lieu voit toujours des workers.
//     CORRECTION : dispersion côtière clippée à la terre (plus de workers en mer).
//   • Chaque wilaya couvre les 15 métiers (cycle i % 15 — vérifié par assert).
//   • Déterministe (RNG mulberry32 à graine fixe) : re-run = mêmes données.
//   • UPSERT (replaceOne) et non create-skip : re-lancer le script RÉPARE les
//     docs seedés obsolètes (ex: créés avant l'ajout de subscriptionActive,
//     qui restaient invisibles à cause du paywall de visibilité).
//   • UIDs fictifs (seed-worker-<wilaya>-<n>) : pas de Firebase Auth possible.
//   • phoneNumber en clair (pas de bidx) : décrypté « pass-through » à la
//     lecture par field-crypto — OK pour des workers de test.
//   • Portfolio : photos DU MÉTIER (chantiers, pièces posées, plats, coupes) —
//     jamais de visages. Un portfolio = la galerie des travaux du worker.
//     Source : profession-media.ts (partagée avec le catalogue des métiers).
//   • Genre réaliste (marché algérien) : bâtiment, plomberie, soudure,
//     mécanique, déménagement… = hommes uniquement. Les femmes travaillent
//     le nettoyage, la cuisine (traiteur), la couture et la coiffure.
//   • Noms ↔ photos garantis alignés : randomuser.me expose le genre DANS
//     l'URL (/portraits/men|women/N.jpg). pravatar ne le permettait pas
//     (le ?u= ne fait que hacher la graine) — d'où les visages de femmes sur
//     des prénoms d'hommes. Vérifié par assert.
// ══════════════════════════════════════════════════════════════════════════════

import mongoose from 'mongoose';

import {
  hash32,
  PROFESSION_KEYS,
  professionWorkPhotos,
} from '../../modules/professions/seeders/profession-media';

// ── Config ────────────────────────────────────────────────────────────────────
const MONGODB_URI =
  process.env['MONGODB_URI'] ??
  'mongodb://khidmeti:khidmeti123@localhost:27017/khidmeti?authSource=admin';

// ── 58 wilayas — coords copiées de khid-app/lib/services/wilaya_manager.dart ──
// Ajout: isCoastal pour clipper la dispersion côté mer.
const WILAYAS: { code: number; name: string; lat: number; lng: number; isCoastal: boolean }[] = [
  { code:  1, name: 'Adrar',               lat: 27.8667, lng: -0.2833, isCoastal: false },
  { code:  2, name: 'Chlef',               lat: 36.1667, lng:  1.3333, isCoastal: true  },
  { code:  3, name: 'Laghouat',            lat: 33.8000, lng:  2.8667, isCoastal: false },
  { code:  4, name: 'Oum El Bouaghi',      lat: 35.8667, lng:  7.1167, isCoastal: false },
  { code:  5, name: 'Batna',               lat: 35.5667, lng:  6.1667, isCoastal: false },
  { code:  6, name: 'Béjaïa',              lat: 36.7500, lng:  5.0833, isCoastal: true  },
  { code:  7, name: 'Biskra',              lat: 34.8500, lng:  5.7333, isCoastal: false },
  { code:  8, name: 'Béchar',              lat: 31.6167, lng: -2.2167, isCoastal: false },
  { code:  9, name: 'Blida',               lat: 36.4833, lng:  2.8333, isCoastal: false },
  { code: 10, name: 'Bouira',              lat: 36.3833, lng:  3.9000, isCoastal: false },
  { code: 11, name: 'Tamanrasset',         lat: 22.7833, lng:  5.5167, isCoastal: false },
  { code: 12, name: 'Tébessa',             lat: 35.4000, lng:  8.1167, isCoastal: false },
  { code: 13, name: 'Tlemcen',             lat: 34.8833, lng: -1.3167, isCoastal: true  },
  { code: 14, name: 'Tiaret',              lat: 35.3708, lng:  1.3228, isCoastal: false },
  { code: 15, name: 'Tizi Ouzou',          lat: 36.7000, lng:  4.0500, isCoastal: false },
  { code: 16, name: 'Alger',               lat: 36.7539, lng:  3.0588, isCoastal: true  },
  { code: 17, name: 'Djelfa',              lat: 34.6667, lng:  3.2500, isCoastal: false },
  { code: 18, name: 'Jijel',               lat: 36.8167, lng:  5.7667, isCoastal: true  },
  { code: 19, name: 'Sétif',               lat: 36.1833, lng:  5.4000, isCoastal: false },
  { code: 20, name: 'Saïda',               lat: 34.8333, lng:  0.1500, isCoastal: false },
  { code: 21, name: 'Skikda',              lat: 36.8667, lng:  6.9000, isCoastal: true  },
  { code: 22, name: 'Sidi Bel Abbès',      lat: 35.2000, lng: -0.6333, isCoastal: false },
  { code: 23, name: 'Annaba',              lat: 36.9000, lng:  7.7667, isCoastal: true  },
  { code: 24, name: 'Guelma',              lat: 36.4667, lng:  7.4333, isCoastal: false },
  { code: 25, name: 'Constantine',         lat: 36.3650, lng:  6.6147, isCoastal: false },
  { code: 26, name: 'Médéa',               lat: 36.2667, lng:  2.7500, isCoastal: false },
  { code: 27, name: 'Mostaganem',          lat: 35.9333, lng:  0.0833, isCoastal: true  },
  { code: 28, name: "M'Sila",              lat: 35.7000, lng:  4.5333, isCoastal: false },
  { code: 29, name: 'Mascara',             lat: 35.3960, lng:  0.1400, isCoastal: false },
  { code: 30, name: 'Ouargla',             lat: 31.9500, lng:  5.3333, isCoastal: false },
  { code: 31, name: 'Oran',                lat: 35.6969, lng: -0.6331, isCoastal: true  },
  { code: 32, name: 'El Bayadh',           lat: 33.6833, lng:  1.0167, isCoastal: false },
  { code: 33, name: 'Illizi',              lat: 26.5000, lng:  8.4667, isCoastal: false },
  { code: 34, name: 'Bordj Bou Arréridj',  lat: 36.0667, lng:  4.7667, isCoastal: false },
  { code: 35, name: 'Boumerdès',           lat: 36.7667, lng:  3.4833, isCoastal: true  },
  { code: 36, name: 'El Tarf',             lat: 36.7667, lng:  8.3167, isCoastal: true  },
  { code: 37, name: 'Tindouf',             lat: 27.6750, lng: -8.1333, isCoastal: false },
  { code: 38, name: 'Tissemsilt',          lat: 35.6000, lng:  1.8167, isCoastal: false },
  { code: 39, name: 'El Oued',             lat: 33.3667, lng:  6.8667, isCoastal: false },
  { code: 40, name: 'Khenchela',           lat: 35.4333, lng:  7.1500, isCoastal: false },
  { code: 41, name: 'Souk Ahras',          lat: 36.2833, lng:  7.9500, isCoastal: false },
  { code: 42, name: 'Tipaza',              lat: 36.5931, lng:  2.4458, isCoastal: true  },
  { code: 43, name: 'Mila',                lat: 36.4500, lng:  6.2667, isCoastal: false },
  { code: 44, name: 'Aïn Defla',           lat: 36.2667, lng:  1.9667, isCoastal: false },
  { code: 45, name: 'Naâma',               lat: 33.2667, lng: -0.3167, isCoastal: false },
  { code: 46, name: 'Aïn Témouchent',      lat: 35.2986, lng: -1.1392, isCoastal: true  },
  { code: 47, name: 'Ghardaïa',            lat: 32.4833, lng:  3.6667, isCoastal: false },
  { code: 48, name: 'Relizane',            lat: 35.7372, lng:  0.5536, isCoastal: false },
  { code: 49, name: 'Timimoun',            lat: 29.2500, lng:  0.2333, isCoastal: false },
  { code: 50, name: 'Bordj Badji Mokhtar', lat: 21.3333, lng:  0.9500, isCoastal: false },
  { code: 51, name: 'Ouled Djellal',       lat: 34.4167, lng:  5.0333, isCoastal: false },
  { code: 52, name: 'Béni Abbès',          lat: 30.1333, lng: -2.1667, isCoastal: false },
  { code: 53, name: 'In Salah',            lat: 27.2000, lng:  2.4667, isCoastal: false },
  { code: 54, name: 'In Guezzam',          lat: 19.5667, lng:  5.7667, isCoastal: false },
  { code: 55, name: 'Touggourt',           lat: 33.1167, lng:  6.0667, isCoastal: false },
  { code: 56, name: 'Djanet',              lat: 24.5500, lng:  9.4833, isCoastal: false },
  { code: 57, name: "El M'Ghair",          lat: 33.9500, lng:  5.9333, isCoastal: false },
  { code: 58, name: 'El Meniaa',           lat: 30.5833, lng:  2.8833, isCoastal: false },
];

// ── Métiers — source unique : profession-media.ts / professions.seeder.ts ─────
const PROFESSIONS = PROFESSION_KEYS;

// ── Part de femmes par métier (réalité du terrain en Algérie) ────────────────
// Les métiers absents de cette table sont 100 % masculins : aucune femme ne
// travaille le bâtiment, la plomberie, la soudure, la mécanique ou le
// déménagement ici. Molette : ajuster une part, ou retirer une clé pour
// rendre le métier exclusivement masculin.
const FEMALE_SHARE: Record<string, number> = {
  cleaner: 0.70, // ménage / nettoyage
  caterer: 0.80, // cuisine, traiteur
  tailor:  0.60, // couture (خياطة)
  barber:  0.30, // coiffure — salons femmes
};

// ── Prénoms algériens genre-alignés ──────────────────────────────────────────
const FIRST_NAMES_MALE = [
  'Karim', 'Farid', 'Mohamed', 'Youcef', 'Amine', 'Rachid', 'Bilal', 'Nabil',
  'Samir', 'Hichem', 'Omar', 'Khaled', 'Sofiane', 'Riad', 'Adel', 'Mourad',
  'Hamza', 'Zaki', 'Ilyes', 'Walid', 'Djamel', 'Fouad', 'Lotfi', 'Nassim',
  'Yacine', 'Redouane', 'Tarik', 'Mehdi', 'Anis', 'Brahim',
];
const FIRST_NAMES_FEMALE = [
  'Fatima', 'Amina', 'Meriem', 'Sarah', 'Leila', 'Nadia', 'Samira', 'Yasmina',
  'Khadija', 'Zahra', 'Salima', 'Dounia', 'Rania', 'Imane', 'Sabrina', 'Noura',
  'Lina', 'Hiba', 'Ines', 'Aya', 'Malika', 'Fatiha', 'Souad', 'Hafida',
  'Nassima', 'Rachida', 'Keltoum', 'Zineb', 'Fadila', 'Halima',
];

const LAST_NAMES = [
  'Benali', 'Boumediene', 'Tlemcani', 'Hadjadj', 'Zerrouk', 'Kaci',
  'Messaoudi', 'Brahimi', 'Bouali', 'Djebari', 'Laid', 'Mansouri',
  'Cherif', 'Belkacem', 'Hamidi', 'Saadi', 'Meziane', 'Bouzid',
  'Ferhat', 'Ghali', 'Toumi', 'Slimani', 'Rahmani', 'Ziani',
];

// Densité : 15 workers/wilaya (≥ 15 ⇒ tous les métiers couverts),
// villes majeures densifiées.
const PER_DEFAULT = 15;
const PER_OVERRIDE: Record<number, number> = {
  16: 24, 31: 24, 25: 18, 23: 18, 19: 18, 9: 16, 6: 16,
};

// ── RNG déterministe (mulberry32) — re-run ⇒ mêmes données ⇒ upserts stables ──
function mulberry32(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), t | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

// ── Géométrie côtière simple : clipper la dispersion côté mer ─────────────────
// Pour les wilayas côtières, la Méditerranée est AU NORD (lat > centre).
// On refléchit les points qui tombent au nord du centre vers le sud.
// C'est une heuristique suffisante (pas de shapefile requis), qui élimine
// 100% des workers "en mer" tout en gardant une dispersion réaliste.
function clampCoastal(lat: number, lng: number, wilaya: typeof WILAYAS[0]): { lat: number; lng: number } {
  if (!wilaya.isCoastal) return { lat, lng };

  // Méditerranée au nord de l'Algérie → si lat > centre, on est en mer.
  // On renvoie le point symétrique par rapport au centre (même distance au sud).
  if (lat > wilaya.lat) {
    return { lat: wilaya.lat - (lat - wilaya.lat), lng };
  }
  return { lat, lng };
}

// ── Mongoose Schemas (minimaux — identiques à user.schema.ts / geo-cell) ──────
const UserSchema = new mongoose.Schema(
  {
    _id:            { type: String, required: true },
    name:           { type: String, required: true },
    email:          { type: String, default: '' },
    phoneNumber:    { type: String, default: '' },
    role:           { type: String, default: 'worker' },
    latitude:       { type: Number, default: null },
    longitude:      { type: Number, default: null },
    wilayaCode:     { type: Number, default: null },
    cellId:         { type: String, default: null },
    geoHash:        { type: String, default: null },
    lastUpdated:    { type: Date,   required: true },
    lastCellUpdate: { type: Date,   default: null },
    profileImageUrl:{ type: String, default: null },
    fcmToken:       { type: String, default: null },
    profession:     { type: String, default: null },
    isOnline:       { type: Boolean, default: false },
    averageRating:  { type: Number, default: 0 },
    ratingCount:    { type: Number, default: 0 },
    ratingSum:      { type: Number, default: 0 },
    jobsCompleted:  { type: Number, default: 0 },
    responseRate:   { type: Number, default: 0.7 },
    lastActiveAt:   { type: Date,   default: null },
    // Paywall de visibilité — findWorkers() filtre subscribedOnly:true.
    subscriptionActive: { type: Boolean, default: false },
    subscriptionUntil:  { type: Date,   default: null },
    subscriptionTier:   { type: String, default: null },
    subscriptionPrice:  { type: Number, default: null },
    dailyQuotaSeconds:  { type: Number, default: null },
    monthlyBidQuota:    { type: Number, default: null },
    searchPriority:     { type: Boolean, default: false },
    // Tier Expert — vue Business (b2bOnly). B2B requiert des docs vérifiés.
    b2bAccess:          { type: Boolean, default: false },
    isVerified:         { type: Boolean, default: false },
    // Portfolio photos (Cloudinary URLs)
    portfolio:       { type: [String], default: [] },
  },
  { collection: 'users', versionKey: false },
);

const GeoCellSchema = new mongoose.Schema(
  {
    _id:            { type: String, required: true },
    wilayaCode:     { type: Number, required: true },
    centerLat:      { type: Number, required: true },
    centerLng:      { type: Number, required: true },
    radius:         { type: Number, default: 5.0 },
    adjacentCellIds:{ type: [String], default: [] },
  },
  { collection: 'geographic_cells', versionKey: false },
);

// ── Helpers géo (identiques à LocationService) ────────────────────────────────
const CELL_PRECISION = 2;

function buildCellId(lat: number, lng: number, wilayaCode: number): string {
  const rLat = +lat.toFixed(CELL_PRECISION);
  const rLng = +lng.toFixed(CELL_PRECISION);
  return `${wilayaCode}_${rLat.toFixed(CELL_PRECISION)}_${rLng.toFixed(CELL_PRECISION)}`;
}

function getAdjacentCellIds(cellId: string): string[] {
  const parts = cellId.split('_');
  if (parts.length !== 3) return [];
  const [wilayaStr, latStr, lngStr] = parts;
  const wilayaCode = parseInt(wilayaStr, 10);
  const lat  = parseFloat(latStr);
  const lng  = parseFloat(lngStr);
  const step = Math.pow(10, -CELL_PRECISION);

  const ids: string[] = [];
  for (let dLat = -1; dLat <= 1; dLat++) {
    for (let dLng = -1; dLng <= 1; dLng++) {
      if (dLat === 0 && dLng === 0) continue;
      const adjLat = +(lat + dLat * step).toFixed(CELL_PRECISION);
      const adjLng = +(lng + dLng * step).toFixed(CELL_PRECISION);
      ids.push(`${wilayaCode}_${adjLat.toFixed(CELL_PRECISION)}_${adjLng.toFixed(CELL_PRECISION)}`);
    }
  }
  return ids;
}

function encodeGeoHash(lat: number, lng: number, precision = 6): string {
  const BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz';
  let hash = '', isEven = true, bit = 0, ch = 0;
  let latMin = -90, latMax = 90, lngMin = -180, lngMax = 180;
  while (hash.length < precision) {
    let mid: number;
    if (isEven) {
      mid = (lngMin + lngMax) / 2;
      if (lng >= mid) { ch |= (1 << (4 - bit)); lngMin = mid; } else { lngMax = mid; }
    } else {
      mid = (latMin + latMax) / 2;
      if (lat >= mid) { ch |= (1 << (4 - bit)); latMin = mid; } else { latMax = mid; }
    }
    isEven = !isEven;
    if (bit < 4) { bit++; } else { hash += BASE32[ch]; bit = 0; ch = 0; }
  }
  return hash;
}

// ── Photo de profil ───────────────────────────────────────────────────────────
// randomuser.me met le genre DANS le chemin (/portraits/men|women/N.jpg) : un
// prénom masculin ne peut donc plus tomber sur un visage de femme. 100 portraits
// par genre (0-99), choisis par hash de l'uid ⇒ stable entre deux re-seeds.
function profilePhoto(uid: string, gender: 'male' | 'female'): string {
  return `https://randomuser.me/api/portraits/${gender === 'male' ? 'men' : 'women'}/${hash32(uid) % 100}.jpg`;
}

// ── Génération des workers ────────────────────────────────────────────────────
interface SeedWorker {
  uid: string; name: string; phone: string; profession: string;
  rating: number; jobs: number; isOnline: boolean; b2bAccess: boolean;
  lat: number; lng: number; wilayaCode: number; wilayaName: string;
  gender: 'male' | 'female';
  portfolioCount: number;
}

export function generateWorkers(): SeedWorker[] {
  const out: SeedWorker[] = [];
  for (const w of WILAYAS) {
    const n    = PER_OVERRIDE[w.code] ?? PER_DEFAULT;
    const rand = mulberry32(w.code * 7919); // graine fixe par wilaya
    for (let i = 0; i < n; i++) {
      // Cycle sur les métiers ⇒ couverture garantie de chaque métier.
      const profession = PROFESSIONS[i % PROFESSIONS.length];

      // Genre : dicté par le métier (0 % de femmes hors nettoyage / cuisine /
      // couture / coiffure), pas par un ratio global.
      const gender = rand() < (FEMALE_SHARE[profession] ?? 0) ? 'female' : 'male';
      const firstNames = gender === 'male' ? FIRST_NAMES_MALE : FIRST_NAMES_FEMALE;
      const firstName = firstNames[Math.floor(rand() * firstNames.length)];
      const lastName  = LAST_NAMES[Math.floor(rand() * LAST_NAMES.length)];

      // Dispersion en 2 anneaux : 1 worker sur 3 reste près du centre
      // (±9 km), les autres couvrent toute la wilaya (±0.35° ≈ ±39 km).
      let lat = +(w.lat + (rand() - 0.5) * (i % 3 === 0 ? 0.16 : 0.70)).toFixed(6);
      let lng = +(w.lng + (rand() - 0.5) * (i % 3 === 0 ? 0.16 : 0.70)).toFixed(6);

      // Clip côtier : renvoie les points "en mer" vers la terre.
      const clamped = clampCoastal(lat, lng, w);
      lat = clamped.lat;
      lng = clamped.lng;

      // Portfolio : 3 à 5 photos (pool Unsplash = 6 par métier, évite débordement)
      const portfolioCount = 3 + Math.floor(rand() * 3);

      out.push({
        uid:        `seed-worker-${w.code}-${String(i + 1).padStart(3, '0')}`,
        name:       `${firstName} ${lastName}`,
        phone:      `+2135${50000000 + w.code * 10000 + i}`,
        profession,
        rating:     +(3.2 + rand() * 1.8).toFixed(1),          // 3.2 → 5.0
        jobs:       3 + Math.floor(rand() * 77),               // 3 → 79
        isOnline:   rand() < 0.85,                             // ~15% hors ligne
        b2bAccess:  rand() < 0.2,                              // ~20% tier Expert
        lat,
        lng,
        wilayaCode: w.code,
        wilayaName: w.name,
        gender,
        portfolioCount,
      });
    }
  }
  return out;
}

// ── Self-check ────────────────────────────────────────────────────────────────
// Tourne avant toute écriture Mongo (donc `make scripts-seed-workers` échoue
// bruyamment plutôt que de seeder des données incohérentes). Exporté pour être
// appelable sans DB depuis un runner TypeScript.
export function selfCheck(workers: SeedWorker[]): void {
  // Chaque wilaya couvre les 15 métiers ET les 2 anneaux de dispersion
  // (au moins un worker à > 0.1° ≈ 11 km du centre).
  // Vérification côtière : aucun worker n'a lat > centre pour wilaya côtière.
  for (const w of WILAYAS) {
    const ws      = workers.filter((x) => x.wilayaCode === w.code);
    const covered = new Set(ws.map((x) => x.profession));
    if (covered.size !== PROFESSIONS.length) {
      throw new Error(`Couverture métiers incomplète — wilaya ${w.code} (${covered.size}/${PROFESSIONS.length})`);
    }
    if (!ws.some((x) => Math.abs(x.lat - w.lat) > 0.1 || Math.abs(x.lng - w.lng) > 0.1)) {
      throw new Error(`Dispersion large absente — wilaya ${w.code} (tous les workers < 11 km du centre)`);
    }
    if (w.isCoastal && ws.some((x) => x.lat > w.lat)) {
      throw new Error(`Worker en mer détecté — wilaya ${w.code} (${w.name})`);
    }
  }

  // Self-check réalisme + cohérence des images (les 3 bugs corrigés) :
  //   1. aucune femme dans un métier de force,
  //   2. prénom ↔ visage : le genre est dans l'URL du portrait,
  //   3. portfolio = travaux du métier, jamais des visages.
  for (const w of workers) {
    if (w.gender === 'female' && !(w.profession in FEMALE_SHARE)) {
      throw new Error(`Femme sur un métier masculin — ${w.uid} (${w.profession})`);
    }
    const photo = profilePhoto(w.uid, w.gender);
    if (!photo.includes(w.gender === 'male' ? '/men/' : '/women/')) {
      throw new Error(`Photo de profil ≠ genre — ${w.uid} (${w.name})`);
    }
    const portfolio = professionWorkPhotos(w.profession, w.uid, w.portfolioCount);
    if (portfolio.length !== w.portfolioCount || new Set(portfolio).size !== portfolio.length) {
      throw new Error(`Portfolio incomplet ou doublons — ${w.uid}`);
    }
    if (portfolio.some((u) => !u.includes('images.unsplash.com/photo-') || /pravatar|portraits/.test(u))) {
      throw new Error(`Portfolio hors métier (ou visages) — ${w.uid} (${w.profession})`);
    }
  }
  // Les femmes existent quand même là où c'est réel (sinon la table est cassée).
  if (!workers.some((w) => w.gender === 'female')) {
    throw new Error('Aucune femme générée — FEMALE_SHARE est cassé');
  }
}

// ── Main ──────────────────────────────────────────────────────────────────────
async function main() {
  const shouldClear = process.argv.includes('--clear');
  const workers     = generateWorkers();
  selfCheck(workers);

  console.log('\n══════════════════════════════════════════════');
  console.log('  Khidmeti — Seed : workers (58 wilayas)');
  console.log('══════════════════════════════════════════════\n');

  await mongoose.connect(MONGODB_URI);
  console.log(`✅ Connecté à MongoDB (${workers.length} workers à seeder)\n`);

  const UserModel    = mongoose.model('User',           UserSchema);
  const GeoCellModel = mongoose.model('GeographicCell', GeoCellSchema);

  const allCellIds = [...new Set(
    workers.map((w) => buildCellId(w.lat, w.lng, w.wilayaCode)),
  )];

  if (shouldClear) {
    const delWorkers = await UserModel.deleteMany({ _id: /^seed-worker-/ });
    const delCells   = await GeoCellModel.deleteMany({ _id: { $in: allCellIds } });
    console.log(`🗑️  ${delWorkers.deletedCount} worker(s) seed supprimés`);
    console.log(`🗑️  ${delCells.deletedCount} cellule(s) seed supprimées\n`);
  }

  // ── Workers : UPSERT (replaceOne) — répare aussi les docs seedés obsolètes ──
  const now = new Date();
  const userOps = workers.map((w) => {
    const cellId  = buildCellId(w.lat, w.lng, w.wilayaCode);
    const geoHash = encodeGeoHash(w.lat, w.lng, 6);

    // Moyenne bayésienne — identique à UsersService.applyRating().
    const ratingSum   = w.rating * w.jobs;
    const C = 3.5, m = 10;
    const bayesianAvg = (m * C + ratingSum) / (m + w.jobs);

    // Photo de profil : genre garanti par l'URL
    const profileUrl = profilePhoto(w.uid, w.gender);

    // Portfolio : photos du MÉTIER (travaux), déterministes par uid
    const portfolio = professionWorkPhotos(w.profession, w.uid, w.portfolioCount);

    return {
      replaceOne: {
        filter: { _id: w.uid },
        replacement: {
          _id:             w.uid,
          name:            w.name,
          email:           '',
          phoneNumber:     w.phone,
          role:            'worker',
          latitude:        w.lat,
          longitude:       w.lng,
          wilayaCode:      w.wilayaCode,
          cellId,
          geoHash,
          lastUpdated:     now,
          lastCellUpdate:  now,
          profileImageUrl: profileUrl,
          fcmToken:        null,
          profession:      w.profession,
          isOnline:        w.isOnline,
          averageRating:   bayesianAvg,
          ratingCount:     w.jobs,
          ratingSum,
          jobsCompleted:   w.jobs,
          responseRate:    0.85,
          lastActiveAt:    now,
          // Abonnement visibilité 1 an — requis par le paywall findWorkers.
          // Entitlements = pack business/expert (illimité, priorité).
          subscriptionActive: true,
          subscriptionUntil:  new Date(Date.now() + 365 * 24 * 60 * 60 * 1000),
          subscriptionTier:   w.b2bAccess ? 'expert' : 'business',
          subscriptionPrice:  w.b2bAccess ? 2500 : 1500,
          dailyQuotaSeconds:  null,
          monthlyBidQuota:    null,
          searchPriority:     true,
          b2bAccess:          w.b2bAccess,
          // Le gate B2B exige des docs vérifiés — cohérent pour les seeds Expert.
          isVerified:         w.b2bAccess,
          // Portfolio photos
          portfolio,
        },
        upsert: true,
      },
    };
  });
  const ur = await UserModel.bulkWrite(userOps, { ordered: false });

  // ── Cellules : upsert $setOnInsert (no-op si déjà présentes) ────────────────
  const cellOps = allCellIds.map((cellId) => {
    const [wilayaStr, latStr, lngStr] = cellId.split('_');
    return {
      updateOne: {
        filter: { _id: cellId },
        update: {
          $setOnInsert: {
            wilayaCode:      parseInt(wilayaStr, 10),
            centerLat:       parseFloat(latStr),
            centerLng:       parseFloat(lngStr),
            radius:          5.0,
            adjacentCellIds: getAdjacentCellIds(cellId),
          },
        },
        upsert: true,
      },
    };
  });
  const cr = await GeoCellModel.bulkWrite(cellOps, { ordered: false });

  // ── Résumé par wilaya ──────────────────────────────────────────────────────
  console.log('  Répartition :');
  for (const w of WILAYAS) {
    const ws     = workers.filter((x) => x.wilayaCode === w.code);
    const online = ws.filter((x) => x.isOnline).length;
    const female = ws.filter((x) => x.gender === 'female').length;
    const male   = ws.length - female;
    const avgPortfolio = (ws.reduce((s, x) => s + x.portfolioCount, 0) / ws.length).toFixed(1);
    console.log(
      `  ${String(w.code).padStart(2)} ${w.name.padEnd(20)} ` +
      `${String(ws.length).padStart(3)} workers | 🟢 ${online} en ligne | ♂ ${male} ♀ ${female} | 📸 ${avgPortfolio} photos moy.`,
    );
  }

  console.log('\n══════════════════════════════════════════════');
  console.log(`  Workers  : ✅ ${ur.upsertedCount} créés | ♻️  ${ur.modifiedCount} mis à jour`);
  console.log(`  Cellules : ✅ ${cr.upsertedCount} créées | ⏭️  ${allCellIds.length - cr.upsertedCount} existantes`);
  console.log('══════════════════════════════════════════════');
  console.log('\n  Tests rapides :');
  console.log('  curl "http://localhost:3000/workers?wilayaCode=31&isOnline=true"');
  console.log('  curl "http://localhost:3000/workers?wilayaCode=16&profession=plumber"\n');

  await mongoose.disconnect();
}

// Lancé comme script (ts-node) ⇒ on seede. Importé (self-check hors DB) ⇒ non.
if (process.argv[1]?.includes('seed-workers')) {
  main().catch((err) => {
    console.error('\n❌ Erreur :', err.message);
    process.exit(1);
  });
}