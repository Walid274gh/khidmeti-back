// ══════════════════════════════════════════════════════════════════════════════
// PROFESSION MEDIA — curated work photos (Unsplash, free licence)
//
// POURQUOI : loremflickr servait des photos aléatoires Flickr avec licences
// inconnues, contenu hors sujet (bus sur "electrician", AC units sans humains),
// et 500 errors. Remplacé par une collection vérifiée à la main montrant des
// travailleurs EN TRAIN DE FAIRE LE MÉTIER, pas juste des outils ou objets.
//
// SOURCE : Unsplash (https://unsplash.com/license) — usage commercial gratuit,
// attribution optionnelle. CDN hotlinking encouragé pour les apps. Les images
// seedées sont DEMO : l'admin remplace via Cloudinary dans le dashboard.
//
// SÉLECTION : 6 photos par métier, eyeballed sur contact sheets (~136 candidats
// → top 6 par profession). Hero = première de la liste (carte catalogue).
// Portfolio workers = rotation déterministe sur les 6 (via professionWorkPhotos).
//
// FORMAT URL : images.unsplash.com/photo-{id}?w=800&q=72&fm=jpg&fit=crop
// ══════════════════════════════════════════════════════════════════════════════

/** Photo IDs Unsplash, ordonnées par préférence (0 = hero). */
export const PROFESSION_PHOTOS: Record<string, string[]> = {
  plumber: [
    '1676210134188-4c05dd172f89', // man working on pipe under sink
    '1676210134190-3f2c0d5cf58d', // fixing water heater
    '1676210134050-6f12c6898395', // red gloves working on toilet
    '1748442001865-5583ec02ae22', // worker inspects pipes and valves
    '1676210133055-eab6ef033ce3', // hands on copper pipe
    '1615749721143-3a38368d3d05', // plumber at sink
  ],
  electrician: [
    '1621905251189-08b45d6a269e', // installing wiring, yellow hardhat
    '1758101755915-462eddc23f57', // testing panel with multimeter
    '1676630656246-3047520adfdf', // working on wall with screwdriver
    '1595856619767-ab739fa7daae', // safety gear on power pole
    '1621905251918-48416bd8575a', // plaid + hard hat holding tool
    '1625148230889-8195e85aae6b', // holding power tool
  ],
  ac_repair: [
    '1737012197886-7d5a52ded45b', // person working on ceiling AC unit
    '1642749776312-aa42ce20c9f5', // two men on roof installation
    '1705579605238-24a90c8799c5', // technician on roof
    '1698998882494-57c3e043f340', // worker on machine
    '1568236700632-c0cfc08f486a', // man fixing gear (grayscale)
    '1748027869634-fc2e545cfb0c', // worker examining equipment
  ],
  mason: [
    '1704005446393-0262b4cb6877', // man working on brick wall with trowel
    '1673865641469-34498379d8af', // blue overalls working cement
    '1704005445445-2747074be8ac', // trowel on brick wall close-up
    '1701850009190-2859ba2aeea6', // applying cement on brick wall
    '1707655536176-a2bc5e00cba9', // working on concrete
    '1743130960579-f88d04835413', // leveling concrete
  ],
  painter: [
    '1688372199140-cade7ae820fe', // man painting wall yellow
    '1717281234297-3def5ae3eee1', // painting wall with roller
    '1693985120993-e9b203ce7631', // close-up roller on wall
    '1776269077163-e26abbc6189d', // painting exterior from ladder
    '1751666526244-40239a251eae', // woman paints wall
    '1715021927612-63269dacb5ea', // man with roller
  ],
  carpenter: [
    '1659930087003-2d64e33181f7', // cutting wood with saw
    '1679797850019-3d0d8659a695', // working on piece of wood
    '1631396326646-c06a935ff3a6', // working on chair in workshop
    '1687422810663-c316494f725a', // smiling, works on wood
    '1505798577917-a65157d3320a', // standing in front of miter saw
    '1608613304899-ea8098577e38', // holding power tool
  ],
  plasterer: [
    '1768839725085-829e6ac7ac26', // hands applying plaster with trowels
    '1761986757577-140af8859587', // man plastering wall with trowel
    '1694521787149-ee0b6a3d9f78', // on ladder working ceiling
    '1701850009190-2859ba2aeea6', // cement on brick wall
    '1685464197644-41d9b07e1e73', // two men orange vests building
    '1694522362256-6c907336af43', // grinder on concrete
  ],
  welder: [
    '1504328345606-18bbc8c9d7d1', // man using welding machine
    '1698664683348-f9f35b809821', // welder working on metal
    '1507497806295-753c4108560c', // wearing auto-dark welding helmet
    '1526634140919-468dc3ae3870', // person welding metal
    '1683470157212-cd4005549fce', // welder on metal piece
    '1714504904786-b6732390b206', // welder in factory
  ],
  cleaner: [
    '1740657254989-42fe9c3b8cce', // yellow gloves cleaning floor
    '1686178827149-6d55c72d81df', // woman vacuuming
    '1713110824336-f78c320dcf8e', // white gloves cleaning chair
    '1742483359033-13315b247c74', // protective suit cleaning carpet
    '1647381518264-97ff1835026f', // woman with broom in kitchen
    '1563453392212-326f5e854473', // holding spray bottle
  ],
  appliance_repair: [
    '1698998882494-57c3e043f340', // working on machine in shop
    '1648815546048-6da4f0083a6d', // working metal object
    '1662767784028-f22926bc7dd8', // man working on machine
    '1635294084898-9a7ed6559eb4', // working on machinery
    '1748640857973-93524ef0fe7d', // smiling machinist in workshop
    '1562941995-17dc31eaaf6d', // man soldering wires
  ],
  mover: [
    '1769972557854-7eae6f95585b', // man carrying mattress
    '1698917414969-feade59e3343', // unloading furniture from truck
    '1694715669993-ea0022b470f7', // unloading boxes from van
    '1682973441491-6b41b7af1c6f', // moving a rug (b&w)
    '1714647211902-bb711d643a17', // woman moving boxes
    '1523543659209-5c57c05834aa', // dragging wagon with boxes
  ],
  mechanic: [
    '1615906655593-ad0386982a0f', // mechanic working on car engine
    '1643700973089-baa86a1ab9ee', // working on car in garage
    '1643701079732-3b1c7a797e3d', // working under vehicle
    '1728474751252-9c085659f6ab', // working on engine with wrench
    '1555140713-973b9f36cd1e', // fixing car daytime
    '1711386689622-1cda23e10217', // car in garage
  ],
  barber: [
    '1647140655214-e4a2d914971f', // cutting hair with scissors
    '1593702275687-f8b402bf1fb5', // white shirt cutting hair
    '1635273051937-a0ddef9573b6', // customer getting haircut
    '1517832606299-7ae9b720a186', // clipping beard (grayscale)
    '1567894340315-735d7c361db0', // man cutting hair
    '1635273051839-003bf06a8751', // close up cutting hair
  ],
  tailor: [
    '1630930678172-63343537a00a', // using sewing machine
    '1606501126768-b78d4569d3f9', // person sewing
    '1641320197434-6ae0ca235048', // close up sewing machine
    '1533758488827-1ed0f9b03899', // person sewing fabric
    '1745847655362-fc86d76584b2', // sewing machine and hands
    '1649182369902-adc2d56cd6c9', // working on sewing machine
  ],
  caterer: [
    '1600565193348-f74bd3c7ccdf', // chef cooking with flames
    '1577219492769-b63a779fac28', // chef preparing foods
    '1681270543584-8e541a1bb056', // chef preparing food
    '1566554273541-37a9ca77b91f', // chef using knife
    '1654922207993-2952fec328ae', // man cooking in kitchen
    '1577106263724-2c8e03bfe9cf', // putting food on plate
  ],
};

/** FNV-1a 32 bits — même que v1, gardé pour la compatibilité signatures. */
export function hash32(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** URL Unsplash CDN optimisée (800px, q72, jpg, crop). */
function unsplashUrl(photoId: string): string {
  return `https://images.unsplash.com/photo-${photoId}?w=800&q=72&fm=jpg&fit=crop`;
}

/** Image hero du métier (carte catalogue) = première de la liste. */
export function professionHero(key: string): string {
  const pool = PROFESSION_PHOTOS[key];
  if (!pool?.length) throw new Error(`profession-media: unknown key "${key}"`);
  return unsplashUrl(pool[0]);
}

/**
 * Portfolio d'un worker : rotation déterministe sur le pool (max 6 photos).
 * Si `n` > pool.length, plafonne à pool.length (évite les doublons).
 */
export function professionWorkPhotos(key: string, seed: string, n: number): string[] {
  const pool = PROFESSION_PHOTOS[key];
  if (!pool?.length) throw new Error(`profession-media: unknown key "${key}"`);
  const cap = Math.min(n, pool.length);
  const start = hash32(seed) % pool.length;
  const out: string[] = [];
  for (let i = 0; i < cap; i++) {
    out.push(unsplashUrl(pool[(start + i) % pool.length]));
  }
  return out;
}

/** Exported pour la validation (remplace PROFESSION_TAGS). */
export const PROFESSION_KEYS = Object.keys(PROFESSION_PHOTOS);
