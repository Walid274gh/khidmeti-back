// ══════════════════════════════════════════════════════════════════════════════
// KHIDMETI — Script de seed : images des professions
//
// USAGE via Makefile (recommandé) :
//   make scripts-seed-professions              ← patch imageUrl (upsert)
//   make scripts-seed-professions ARGS=--clear ← remet les URLs par défaut
//
// PROPRIÉTÉS :
//   • Upsert idempotent : relancer le script n'écrase pas les photos admin.
//     Si imageUrl est déjà définie sur un doc, on ne le touche PAS.
//   • --clear  : force $set même si imageUrl est déjà présente (réinitialise
//     vers les images par défaut — pratique après un test admin foireux).
//   • Les URLs viennent de profession-media.ts (recherche par tag, photos
//     réelles du métier) — source unique partagée avec professions.seeder.ts
//     et seed-workers.ts. Plus d'ID photo deviné, donc plus d'image hors sujet.
//     L'admin peut les remplacer individuellement via le dashboard web.
// ══════════════════════════════════════════════════════════════════════════════

import mongoose from 'mongoose';

import { PROFESSION_KEYS, professionHero } from '../../modules/professions/seeders/profession-media';

// ── Config ────────────────────────────────────────────────────────────────────
const MONGODB_URI =
  process.env['MONGODB_URI'] ??
  'mongodb://khidmeti:khidmeti123@localhost:27017/khidmeti?authSource=admin';

const FORCE_CLEAR = process.argv.includes('--clear');

// ── Schéma minimal (imageUrl seulement) ──────────────────────────────────────
const ProfessionSchema = new mongoose.Schema({}, { strict: false });
const ProfessionModel  = mongoose.model('Profession', ProfessionSchema, 'professions');

// ── Main ──────────────────────────────────────────────────────────────────────
async function main() {
  console.log('\n══════════════════════════════════════════════');
  console.log('  Khidmeti — seed-professions : images');
  console.log(`  Mode : ${FORCE_CLEAR ? '⚠️  --clear (écrase imageUrl existantes)' : 'normal (skip si imageUrl présente)'}`);
  console.log('══════════════════════════════════════════════\n');

  await mongoose.connect(MONGODB_URI);
  console.log('  ✅ MongoDB connecté\n');

  let updated = 0;
  let skipped = 0;

  for (const key of PROFESSION_KEYS) {
    const imageUrl = professionHero(key);
    const filter = FORCE_CLEAR
      ? { key }
      : { key, $or: [{ imageUrl: { $exists: false } }, { imageUrl: null }, { imageUrl: '' }] };

    const result = await (ProfessionModel as any).updateOne(
      filter,
      { $set: { imageUrl } },
    );

    if (result.modifiedCount > 0) {
      console.log(`  ✅  ${key.padEnd(20)} → ${imageUrl}`);
      updated++;
    } else if (result.matchedCount === 0 && !FORCE_CLEAR) {
      // Doc non trouvé ou imageUrl déjà renseignée
      const exists = await (ProfessionModel as any).exists({ key });
      if (exists) {
        console.log(`  ⏭️  ${key.padEnd(20)} → imageUrl déjà présente (skip)`);
        skipped++;
      } else {
        console.log(`  ⚠️  ${key.padEnd(20)} → profession introuvable en base`);
      }
    } else {
      console.log(`  ⏭️  ${key.padEnd(20)} → inchangé`);
      skipped++;
    }
  }

  console.log('\n══════════════════════════════════════════════');
  console.log(`  Images : ✅ ${updated} définies | ⏭️  ${skipped} ignorées`);
  console.log('══════════════════════════════════════════════');
  console.log('\n  Astuce : pour remplacer par vos propres images, utilisez');
  console.log('  le dashboard admin → Professions → ✏️ Modifier → Photo.\n');

  await mongoose.disconnect();
}

main().catch((err) => {
  console.error('\n❌ Erreur :', err.message);
  process.exit(1);
});
