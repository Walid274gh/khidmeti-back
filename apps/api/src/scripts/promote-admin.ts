// ══════════════════════════════════════════════════════════════════════════════
// KHIDMETI — Promote a user to the `admin` role
//
// Creates the first (or any) admin able to sign into the web dashboard. It:
//   1. Finds the user in Mongo by email OR uid and sets role = 'admin'.
//   2. (Best-effort) sets a Firebase custom claim { admin: true } on that uid.
//
// The DB role is authoritative for AdminGuard; the claim is a convenience for
// any future client-side gating.
//
// USAGE via Makefile (recommended — runs inside the api container):
//   make scripts-promote-admin ARGS="--email you@example.com"
//   make scripts-promote-admin ARGS="--uid <firebase-uid>"
//
// USAGE direct:
//   npx ts-node --project tsconfig.json src/scripts/promote-admin.ts --email you@example.com
//
// PREREQUISITE: the account must already exist in Firebase Auth (sign up once
// via the web dashboard login with a password, or in the Firebase console) and
// have a profile document in Mongo. If no Mongo profile exists yet, pass --uid
// and a minimal profile is created.
// ══════════════════════════════════════════════════════════════════════════════

import mongoose from 'mongoose';
import * as admin from 'firebase-admin';
import { blindIndex, encryptField } from '../common/crypto/field-crypto';

const MONGODB_URI =
  process.env['MONGODB_URI'] ??
  'mongodb://khidmeti:khidmeti123@localhost:27017/khidmeti?authSource=admin';

function parseArgs(): { email?: string; uid?: string; name?: string } {
  const args = process.argv.slice(2);
  const out: { email?: string; uid?: string; name?: string } = {};
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--email') out.email = args[++i];
    else if (a === '--uid') out.uid = args[++i];
    else if (a === '--name') out.name = args[++i];
  }
  return out;
}

function initFirebase(): boolean {
  if (admin.apps.length > 0) return true;
  const projectId = process.env['FIREBASE_PROJECT_ID'];
  const clientEmail = process.env['FIREBASE_CLIENT_EMAIL'];
  const privateKey = process.env['FIREBASE_PRIVATE_KEY']?.replace(/\\n/g, '\n');
  if (!projectId || !clientEmail || !privateKey) {
    console.warn('⚠️  Firebase credentials missing — will skip setting custom claim.');
    return false;
  }
  admin.initializeApp({
    credential: admin.credential.cert({ projectId, clientEmail, privateKey }),
  });
  return true;
}

async function main(): Promise<void> {
  const { email, uid, name } = parseArgs();
  if (!email && !uid) {
    console.error('❌  Provide --email <email> or --uid <firebase-uid>.');
    process.exit(1);
  }

  const hasFirebase = initFirebase();

  await mongoose.connect(MONGODB_URI);
  const users = mongoose.connection.collection<{ _id: string; [key: string]: unknown }>('users');

  // Resolve the target document. Emails are stored AES-256-GCM-encrypted since
  // the PII hardening, so equality lookups go through the blind index; the
  // plaintext fallback only matches rows the backfill has not touched yet.
  const normEmail = email?.toLowerCase().trim();
  let query: Record<string, unknown> | null = null;
  if (uid) query = { _id: uid };
  else if (normEmail)
    query = { $or: [{ emailBidx: blindIndex(normEmail) }, { email: normEmail }] };

  let doc = query ? await users.findOne(query) : null;

  // Email given but no Mongo profile (console-created Firebase accounts have
  // none): resolve the uid via Firebase Auth so the minimal profile below can
  // be created without asking the operator to copy the uid by hand.
  let effectiveUid = uid;
  if (!doc && !effectiveUid && normEmail && hasFirebase) {
    try {
      effectiveUid = (await admin.auth().getUserByEmail(normEmail)).uid;
      console.log(`ℹ️  Firebase: ${normEmail} → uid=${effectiveUid}`);
    } catch {
      // Falls through to the "no user found" error below.
    }
  }

  // If no profile exists, create a minimal admin profile.
  if (!doc && effectiveUid) {
    const now = new Date();
    await users.insertOne({
      _id: effectiveUid,
      name: name ?? 'Administrator',
      // Same at-rest shape the encryptedFields plugin writes.
      email: encryptField(normEmail ?? ''),
      emailBidx: blindIndex(normEmail ?? ''),
      phoneNumber: '',
      phoneNumberBidx: '',
      role: 'admin',
      isBanned: false,
      isVerified: true,
      latitude: null,
      longitude: null,
      lastUpdated: now,
      language: 'fr',
      profileImageUrl: null,
      fcmToken: null,
      profession: null,
      isOnline: false,
      averageRating: 0,
      ratingCount: 0,
      ratingSum: 0,
      jobsCompleted: 0,
      responseRate: 0.7,
      lastActiveAt: null,
    });
    doc = await users.findOne({ _id: effectiveUid });
    console.log(`✅  Created new admin profile for uid=${effectiveUid}`);
  }

  if (!doc) {
    console.error(
      `❌  No user found for ${email ? `email=${email}` : `uid=${uid}`}. ` +
        `Sign in once via the dashboard (to create the Firebase + Mongo record) then retry, ` +
        `or pass --uid to create a profile directly.`,
    );
    await mongoose.disconnect();
    process.exit(1);
  }

  const targetUid = doc._id as string;

  await users.updateOne({ _id: targetUid }, { $set: { role: 'admin', isBanned: false } });
  console.log(`✅  Mongo: user ${targetUid} role set to 'admin'.`);

  if (hasFirebase) {
    try {
      await admin.auth().setCustomUserClaims(targetUid, { admin: true });
      console.log(`✅  Firebase: custom claim { admin: true } set on ${targetUid}.`);
    } catch (err) {
      console.warn(`⚠️  Could not set Firebase claim: ${(err as Error).message}`);
    }
  }

  console.log('\n🎉  Done. This account can now sign into the /admin dashboard.\n');
  await mongoose.disconnect();
}

main().catch(async (err) => {
  console.error('\n❌  Error:', err.message);
  await mongoose.disconnect().catch(() => undefined);
  process.exit(1);
});
