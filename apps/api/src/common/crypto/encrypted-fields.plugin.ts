// ══════════════════════════════════════════════════════════════════════════════
// Mongoose plugin — transparent field encryption + blind index.
//
// For each configured field `f`:
//   • writes (save / insertMany / updateOne / updateMany / findOneAndUpdate):
//       f      → AES-256-GCM ciphertext
//       f+Bidx → HMAC blind index (deterministic, for uniqueness + exact search)
//   • reads (find / findOne / findById / findOneAndUpdate returning the doc):
//       f      → decrypted plaintext (via the 'init' document hook)
//
// .lean() reads bypass the 'init' hook and therefore return ciphertext — callers
// that use .lean() (admin.service) must decrypt explicitly. This is intentional:
// lean is a raw-BSON escape hatch and the plugin cannot hydrate it.
// ══════════════════════════════════════════════════════════════════════════════

import { Schema } from 'mongoose';
import { blindIndex, decryptField, encryptField } from './field-crypto';

export interface EncryptedFieldsOptions {
  /** Plaintext field names to encrypt. A `<field>Bidx` companion is populated. */
  fields: string[];
}

const bidxKey = (f: string): string => `${f}Bidx`;

export function encryptedFields(schema: Schema, opts: EncryptedFieldsOptions): void {
  const { fields } = opts;

  // Encrypt an update payload in place (handles both top-level and $set styles).
  const encryptUpdate = (update: Record<string, unknown>): void => {
    if (!update) return;
    const containers = [update, (update['$set'] as Record<string, unknown>) ?? null].filter(Boolean);
    for (const c of containers as Record<string, unknown>[]) {
      for (const f of fields) {
        if (Object.prototype.hasOwnProperty.call(c, f)) {
          const plain = c[f] as string;
          c[f] = encryptField(plain);
          c[bidxKey(f)] = blindIndex(plain);
        }
      }
    }
  };

  // ── Writes ──────────────────────────────────────────────────────────────────
  schema.pre('save', function (next) {
    for (const f of fields) {
      if (this.isModified(f)) {
        const plain = this.get(f) as string;
        this.set(f, encryptField(plain));
        this.set(bidxKey(f), blindIndex(plain));
      }
    }
    next();
  });

  schema.pre('insertMany', function (next, docs: Record<string, unknown>[]) {
    for (const doc of docs) {
      for (const f of fields) {
        if (Object.prototype.hasOwnProperty.call(doc, f)) {
          const plain = doc[f] as string;
          doc[f] = encryptField(plain);
          doc[bidxKey(f)] = blindIndex(plain);
        }
      }
    }
    next();
  });

  // findByIdAndUpdate routes through the findOneAndUpdate hook, so upserts are
  // covered too.
  schema.pre(['updateOne', 'updateMany', 'findOneAndUpdate'], function (next) {
    encryptUpdate(this.getUpdate() as Record<string, unknown>);
    next();
  });

  // ── Reads ───────────────────────────────────────────────────────────────────
  // 'init' fires for every hydrated document (find/findOne/findById and the doc
  // returned by findOneAndUpdate). Not for .lean().
  schema.post('init', function (this: Record<string, unknown>) {
    for (const f of fields) {
      const v = this[f];
      if (typeof v === 'string') this[f] = decryptField(v);
    }
  });
}

/** Decrypt configured fields on a plain (lean) object. For .lean() call sites. */
export function decryptLean<T extends Record<string, unknown>>(obj: T | null, fields: string[]): T | null {
  if (!obj) return obj;
  for (const f of fields) {
    if (typeof obj[f] === 'string') (obj as Record<string, unknown>)[f] = decryptField(obj[f] as string);
  }
  return obj;
}

/** PII fields encrypted on the users collection. Single source of truth. */
export const USER_ENCRYPTED_FIELDS = ['email', 'phoneNumber'] as const;
