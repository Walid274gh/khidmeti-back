// ══════════════════════════════════════════════════════════════════════════════
// PIN hashing — scrypt (node stdlib), timing-safe verify.
//
// Why scrypt and not bcrypt/argon2: both would be new dependencies; scrypt is
// in node:crypto, memory-hard, and OWASP-approved for password storage. A PIN
// is 6 digits (1M combinations) so the real defense is the server-side attempt
// lock in PinService — the hash only has to make an offline dump non-trivial.
//
// Format: `s1:<salt hex>:<hash hex>` — versioned like field-crypto's `v1:`.
// ══════════════════════════════════════════════════════════════════════════════

import { randomBytes, scryptSync, timingSafeEqual } from 'crypto';

const VERSION  = 's1';
const SALT_LEN = 16;
const KEY_LEN  = 32;
// N=2^15, r=8, p=1 — ~50ms on server hardware. Interactive-login cost class.
const PARAMS = { N: 2 ** 15, r: 8, p: 1, maxmem: 64 * 1024 * 1024 };

export function hashPin(pin: string): string {
  const salt = randomBytes(SALT_LEN);
  const hash = scryptSync(pin, salt, KEY_LEN, PARAMS);
  return `${VERSION}:${salt.toString('hex')}:${hash.toString('hex')}`;
}

export function verifyPin(pin: string, stored: string): boolean {
  if (!stored || !stored.startsWith(`${VERSION}:`)) return false;
  const [, saltHex, hashHex] = stored.split(':');
  if (!saltHex || !hashHex) return false;
  try {
    const expected = Buffer.from(hashHex, 'hex');
    // Exact length required: Buffer.from('zz','hex') silently yields an EMPTY
    // buffer, and scrypt(keylen=0) === empty → timingSafeEqual(∅,∅) is TRUE.
    // Without this check a corrupt stored hash would verify ANY pin.
    if (expected.length !== KEY_LEN) return false;
    const actual = scryptSync(pin, Buffer.from(saltHex, 'hex'), KEY_LEN, PARAMS);
    return timingSafeEqual(actual, expected);
  } catch {
    return false;
  }
}

// ── Runnable self-check: `node dist/common/crypto/pin-hash.js` ────────────────
if (require.main === module) {
  const assert: typeof import('assert') = require('assert');

  const h = hashPin('123456');
  assert.ok(h.startsWith('s1:'), 'hash is versioned');
  assert.ok(verifyPin('123456', h), 'correct PIN verifies');
  assert.ok(!verifyPin('123457', h), 'wrong PIN rejected');
  assert.ok(!verifyPin('123456', ''), 'empty stored hash rejected');
  assert.ok(!verifyPin('123456', 'garbage'), 'malformed stored hash rejected');
  // Regression: 'zz' hex-decodes to an EMPTY buffer; without the length check
  // scrypt(keylen=0) === empty and timingSafeEqual(∅,∅) verifies ANY pin.
  assert.ok(!verifyPin('123456', 's1:zz:zz'), 'corrupt hex rejected');
  assert.notStrictEqual(hashPin('123456'), hashPin('123456'), 'salt randomizes hash');

  console.log('✓ pin-hash self-check passed');
}
