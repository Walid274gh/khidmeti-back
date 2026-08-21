// ══════════════════════════════════════════════════════════════════════════════
// One-off migration: encrypt existing plaintext email/phoneNumber + fill blind
// indexes on the users collection, then service_requests.userPhone likewise.
//
// RUN ONCE, BEFORE dropping the old email_1 / phoneNumber_1 unique indexes and
// before the new bidx unique indexes are built (see user.schema.ts migration
// note). Idempotent: already-encrypted rows are skipped, so re-running is safe.
//
//   FIELD_ENC_KEY=... FIELD_ENC_PEPPER=... MONGODB_URI=... \
//     node dist/scripts/encrypt-existing-pii.js
//
// Uses the native driver (not the Mongoose model) to bypass the encryption
// plugin — we control encryption explicitly here and stay idempotent.
// ══════════════════════════════════════════════════════════════════════════════

import mongoose from 'mongoose';
import type { AnyBulkWriteOperation } from 'mongodb';
import { blindIndex, encryptField, isEncrypted } from '../common/crypto/field-crypto';

type UserRaw = { _id: string; email?: string; phoneNumber?: string };

async function main(): Promise<void> {
  const uri = process.env['MONGODB_URI'];
  if (!uri) throw new Error('MONGODB_URI is required');
  // Fail fast if the key/pepper are missing before we touch any data.
  blindIndex('warmup');

  await mongoose.connect(uri);
  const coll = mongoose.connection.collection<UserRaw>('users');
  const cursor = coll.find({}, { projection: { email: 1, phoneNumber: 1 } });

  let scanned = 0;
  let updated = 0;
  let ops: AnyBulkWriteOperation<UserRaw>[] = [];

  const flush = async (): Promise<void> => {
    if (ops.length === 0) return;
    await coll.bulkWrite(ops);
    updated += ops.length;
    ops = [];
  };

  for await (const doc of cursor) {
    scanned++;
    const set: Record<string, string> = {};
    if (doc.email && !isEncrypted(doc.email)) {
      set['email'] = encryptField(doc.email);
      set['emailBidx'] = blindIndex(doc.email);
    }
    if (doc.phoneNumber && !isEncrypted(doc.phoneNumber)) {
      set['phoneNumber'] = encryptField(doc.phoneNumber);
      set['phoneNumberBidx'] = blindIndex(doc.phoneNumber);
    }
    if (Object.keys(set).length > 0) {
      ops.push({ updateOne: { filter: { _id: doc._id }, update: { $set: set } } });
      if (ops.length >= 500) await flush();
    }
  }
  await flush();

  console.log(`✓ users done — scanned ${scanned}, encrypted ${updated}`);

  // ── service_requests.userPhone (same treatment, added 2026-08) ──────────────
  const reqColl = mongoose.connection.collection<{ _id: string; userPhone?: string }>(
    'service_requests',
  );
  const reqCursor = reqColl.find({}, { projection: { userPhone: 1 } });
  let reqScanned = 0;
  let reqOps: AnyBulkWriteOperation<{ _id: string; userPhone?: string }>[] = [];
  let reqUpdated = 0;
  const reqFlush = async (): Promise<void> => {
    if (reqOps.length === 0) return;
    await reqColl.bulkWrite(reqOps);
    reqUpdated += reqOps.length;
    reqOps = [];
  };
  for await (const doc of reqCursor) {
    reqScanned++;
    if (doc.userPhone && !isEncrypted(doc.userPhone)) {
      reqOps.push({
        updateOne: {
          filter: { _id: doc._id },
          update: {
            $set: {
              userPhone: encryptField(doc.userPhone),
              userPhoneBidx: blindIndex(doc.userPhone),
            },
          },
        },
      });
      if (reqOps.length >= 500) await reqFlush();
    }
  }
  await reqFlush();
  console.log(`✓ service_requests done — scanned ${reqScanned}, encrypted ${reqUpdated}`);

  await mongoose.disconnect();
}

main().catch((err) => {
  console.error('migration failed:', err);
  process.exit(1);
});
