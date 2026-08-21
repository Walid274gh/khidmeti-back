// ══════════════════════════════════════════════════════════════════════════════
// Field-level crypto — AES-256-GCM encryption + HMAC-SHA256 blind index.
//
// Used to encrypt PII (email, phoneNumber) at the application layer so a full
// database dump never reveals plaintext. This is defense-in-depth ON TOP of
// MongoDB WiredTiger encryption-at-rest (see docker/mongo-encryption-at-rest.md).
//
// SEARCH: ciphertext is non-deterministic (random IV) so it cannot be searched.
// For fields that must stay uniquely-constrained / exact-match searchable we
// store a deterministic *blind index* alongside: HMAC-SHA256(pepper, value).
// Equality lookups query the blind index; substring/regex search is impossible
// on encrypted fields — that is a fundamental property, not a limitation to fix.
//
// KEY MANAGEMENT — CRITICAL:
//   FIELD_ENC_KEY (64 hex chars = 32 bytes) and FIELD_ENC_PEPPER live in env /
//   secret manager, never in code. LOSING THE KEY = LOSING ALL ENCRYPTED PII,
//   irreversibly. Rotate by re-encrypting (add a v2 key path); never edit v1.
// ══════════════════════════════════════════════════════════════════════════════

import { createCipheriv, createDecipheriv, createHmac, randomBytes } from 'crypto';

const VERSION = 'v1';
const IV_LEN = 12;   // GCM standard nonce length
const TAG_LEN = 16;  // GCM auth tag length

let cachedKey: Buffer | null = null;
let cachedPepper: string | null = null;

function key(): Buffer {
  if (cachedKey) return cachedKey;
  const hex = process.env['FIELD_ENC_KEY'];
  if (!hex || hex.length !== 64) {
    throw new Error('FIELD_ENC_KEY must be 64 hex chars (32 bytes). Generate: openssl rand -hex 32');
  }
  cachedKey = Buffer.from(hex, 'hex');
  return cachedKey;
}

function pepper(): string {
  if (cachedPepper !== null) return cachedPepper;
  const p = process.env['FIELD_ENC_PEPPER'];
  if (!p) throw new Error('FIELD_ENC_PEPPER is required. Generate: openssl rand -hex 32');
  cachedPepper = p;
  return cachedPepper;
}

/**
 * Encrypt a UTF-8 string with AES-256-GCM. Returns `v1:<base64(iv|tag|ct)>`.
 * Empty strings pass through unchanged (keeps the "'' = no value" convention
 * and the partial unique index semantics on the blind-index field).
 */
export function encryptField(plain: string): string {
  if (!plain) return '';
  const iv = randomBytes(IV_LEN);
  const cipher = createCipheriv('aes-256-gcm', key(), iv);
  const ct = Buffer.concat([cipher.update(plain, 'utf8'), cipher.final()]);
  const tag = cipher.getAuthTag();
  return `${VERSION}:${Buffer.concat([iv, tag, ct]).toString('base64')}`;
}

/**
 * Decrypt a value produced by encryptField. Values without the `v1:` prefix are
 * assumed still-plaintext (legacy / mid-migration) and returned as-is, so the
 * app keeps working while the migration backfills existing rows.
 *
 * A corrupt/foreign-key ciphertext returns '' instead of throwing: this runs in
 * the Mongoose post('init') hook for EVERY hydrated document, so one bad row
 * must degrade to an empty field, not 500 every list/search that touches it.
 */
export function decryptField(blob: string): string {
  if (!blob || !blob.startsWith(`${VERSION}:`)) return blob ?? '';
  try {
    const raw = Buffer.from(blob.slice(VERSION.length + 1), 'base64');
    const iv = raw.subarray(0, IV_LEN);
    const tag = raw.subarray(IV_LEN, IV_LEN + TAG_LEN);
    const ct = raw.subarray(IV_LEN + TAG_LEN);
    const decipher = createDecipheriv('aes-256-gcm', key(), iv);
    decipher.setAuthTag(tag);
    return Buffer.concat([decipher.update(ct), decipher.final()]).toString('utf8');
  } catch {
    // eslint-disable-next-line no-console
    console.warn('decryptField: GCM auth failed (wrong key or corrupt data) — returning empty');
    return '';
  }
}

/** Is this value already encrypted? Used by the migration to stay idempotent. */
export function isEncrypted(blob: string): boolean {
  return typeof blob === 'string' && blob.startsWith(`${VERSION}:`);
}

/**
 * Deterministic blind index for equality lookups + uniqueness on an encrypted
 * field. Normalizes (trim + lowercase) so 'A@x.com' and ' a@x.com ' collide the
 * same way the old `lowercase: true, trim: true` prop did. Empty → '' (excluded
 * from the partial unique index, matching the old email/phone '' convention).
 */
export function blindIndex(value: string): string {
  const norm = (value ?? '').trim().toLowerCase();
  if (!norm) return '';
  return createHmac('sha256', pepper()).update(norm).digest('hex');
}

// ── Runnable self-check: `node dist/common/crypto/field-crypto.js` ────────────
// Verifies GCM round-trip, tamper detection, blind-index determinism, and the
// empty/plaintext pass-throughs. No test framework — assert only.
if (require.main === module) {
  const assert: typeof import('assert') = require('assert');
  process.env['FIELD_ENC_KEY'] = 'a'.repeat(64);
  process.env['FIELD_ENC_PEPPER'] = 'test-pepper';
  cachedKey = null; cachedPepper = null;

  const msg = 'rachid+test@example.com';
  const enc = encryptField(msg);
  assert.ok(enc.startsWith('v1:') && enc !== msg, 'ciphertext must be prefixed and differ');
  assert.strictEqual(decryptField(enc), msg, 'round-trip must recover plaintext');
  assert.notStrictEqual(encryptField(msg), encryptField(msg), 'IV must randomize ciphertext');

  assert.strictEqual(encryptField(''), '', 'empty encrypts to empty');
  assert.strictEqual(decryptField(''), '', 'empty decrypts to empty');
  assert.strictEqual(decryptField('legacy-plaintext'), 'legacy-plaintext', 'plaintext passes through');

  assert.strictEqual(blindIndex('A@X.com '), blindIndex('a@x.com'), 'blind index normalizes');
  assert.notStrictEqual(blindIndex('a@x.com'), blindIndex('b@x.com'), 'distinct values → distinct index');
  assert.strictEqual(blindIndex(''), '', 'empty blind index is empty');

  const tampered = 'v1:' + Buffer.from('x'.repeat(40)).toString('base64');
  assert.strictEqual(decryptField(tampered), '', 'tampered ciphertext degrades to empty, never throws');

  console.log('✓ field-crypto self-check passed');
}
