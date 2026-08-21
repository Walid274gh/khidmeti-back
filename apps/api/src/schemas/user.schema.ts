// ══════════════════════════════════════════════════════════════════════════════
// User Schema — Unified collection for ALL users (clients & workers)
//
// DESIGN RATIONALE:
//   Every worker is a user — there is no scenario where a worker exists without
//   a user identity. Maintaining two collections (users + workers) duplicated
//   identity fields (name, email, phone, location, fcmToken, profileImageUrl)
//   and forced every service to manage two write paths, two cache entries, and
//   two transaction legs for every auth operation.
//
//   The unified design:
//     • Eliminates duplication — one document per person, always.
//     • Simplifies auth: registration, login, and profile update touch one doc.
//     • `role: 'client' | 'worker'` is the single discriminator.
//     • Worker-specific fields (profession, isOnline, rating…) default to
//       neutral values so client queries never see "online" workers and vice-versa.
//     • Partial indexes restrict heavy worker indexes to role='worker' documents,
//       keeping the index footprint proportional to the actual worker count.
//
// PHONE AUTH — index email :
//   Les utilisateurs authentifiés par téléphone peuvent n'avoir aucun email.
//   L'index email est désormais partiel (partialFilterExpression: email ≠ '')
//   pour ne contraindre l'unicité que sur les emails non vides.
//   Cela évite les collisions entre N utilisateurs avec email = ''.
//
// MIGRATION (run once against production):
//   -- Supprimer l'ancien index unique non-partiel :
//   db.users.dropIndex("email_1");
//
//   -- Créer le nouvel index email partiel :
//   -- (PAS de `sparse` — MongoDB interdit sparse + partialFilterExpression)
//   db.users.createIndex(
//     { email: 1 },
//     { unique: true, partialFilterExpression: { email: { $ne: '' } } }
//   );
//
//   -- Créer l'index phoneNumber partiel :
//   db.users.createIndex(
//     { phoneNumber: 1 },
//     { unique: true, partialFilterExpression: { phoneNumber: { $ne: '' } } }
//   );
//
//   -- Migrer les workers :
//   db.workers.find().forEach(w => {
//     w.role = 'worker';
//     db.users.updateOne({ _id: w._id }, { $set: w }, { upsert: true });
//   });
//   db.workers.drop();
// ══════════════════════════════════════════════════════════════════════════════

import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document } from 'mongoose';
import { encryptedFields, USER_ENCRYPTED_FIELDS } from '../common/crypto/encrypted-fields.plugin';

export enum UserRole {
  Client   = 'client',
  Worker   = 'worker',
  Admin    = 'admin',
  // B2B account (memoire §B2B): behaves like a client, but discovery is
  // filtered to Expert-tier (b2bAccess) workers only.
  Business = 'business',
}

// Subscription packs (business model v2): the protected bid channel is the
// real product, map visibility is the secondary one. All packs are 7/7 —
// the bid quota is the differentiator, not calendar days. Four named presets
// plus a slider-built custom pack:
//   basic    500  — map 5 h/day, NO bids (sees direct requests)
//   pro     1000  — map 10 h/day, 20 bids/month
//   business 1500 — unlimited map + bids, search priority, Pro badge
//   expert   2500 — everything business + B2B flux (requires verified docs)
//   custom 500–2550 — sliders (hours 5–15, bids 0–30 at 25 DZD/unit) plus
//   priority (+200) and B2B (+850, docs required) toggles. Full custom costs
//   more than the Expert preset — presets stay the discounted bundles.
export const SUBSCRIPTION_TIERS = ['basic', 'pro', 'business', 'expert', 'custom'] as const;
export type SubscriptionTier = (typeof SUBSCRIPTION_TIERS)[number];

/** Entitlements written on the user doc at activation — the doc is
 *  self-describing, so the visibility/bid gates never look up tier tables. */
export interface PackEntitlements {
  price: number;                    // DZD/month
  dailyQuotaSeconds: number | null; // null = unlimited map time
  monthlyBidQuota: number | null;   // null = unlimited bids; 0 = no bid access
  searchPriority: boolean;          // ranking boost + Pro badge
  b2bAccess: boolean;               // B2B flux (requires verified docs)
}

export const TIER_PACKS: Record<Exclude<SubscriptionTier, 'custom'>, PackEntitlements> = {
  basic:    { price: 500,  dailyQuotaSeconds: 5 * 3600,  monthlyBidQuota: 0,    searchPriority: false, b2bAccess: false },
  pro:      { price: 1000, dailyQuotaSeconds: 10 * 3600, monthlyBidQuota: 20,   searchPriority: false, b2bAccess: false },
  business: { price: 1500, dailyQuotaSeconds: null,      monthlyBidQuota: null, searchPriority: true,  b2bAccess: false },
  expert:   { price: 2500, dailyQuotaSeconds: null,      monthlyBidQuota: null, searchPriority: true,  b2bAccess: true  },
};

/**
 * Portfolio slots (photos of past work) granted by a pack, derived from its
 * price: 100 DZD = 1 photo. basic 500→5, pro 1000→10, business 1500→15,
 * expert 2500→25, custom 500–2550→5–26 (a fully-loaded custom outgrows the
 * Expert preset here too, exactly as it does on price).
 *
 * WHY DERIVED AND NOT A PAID ADD-ON: the product sells VISIBILITY (map time +
 * bid channel). A portfolio is what converts a visible worker into a hired one,
 * and it serves the CLIENT side, which pays nothing — locking it behind its own
 * price would depress the conversion the subscription is bought for. So every
 * subscribed worker gets one; the pack decides how big it is.
 */
/**
 * Portfolio thumbnails shipped with a DISCOVERY result (search / map cell).
 * Discovery returns up to 50 worker documents at a time, so the full gallery is
 * $slice'd down to what the preview sheet actually draws; GET /workers/:id
 * serves the complete list for the profile page.
 */
export const PORTFOLIO_PREVIEW_COUNT = 3;

export function portfolioQuotaForPrice(price: number): number {
  if (!Number.isFinite(price) || price <= 0) return 0;
  return Math.round(price / 100);
}

// Custom pack pricing: 500 base (5 h/day, 0 bids) + 25/extra hour + 25/bid,
// plus flat add-ons: priority +200, B2B +850.
//   floor (5 h, 0 bids, no add-ons)            = 500
//   sliders maxed (15 h, 30 bids)              = 1500
//   everything on (15 h + 30 bids + both)      = 2550 > Expert preset 2500 —
//   presets stay attractive as discounted bundles (Pro à la carte would be 1125).
export const CUSTOM_PACK = {
  basePrice: 500, hourPrice: 25, bidPrice: 25,
  priorityPrice: 200, b2bPrice: 850,
  hoursMin: 5, hoursMax: 15, bidsMin: 0, bidsMax: 30,
} as const;

/** Build custom-pack entitlements; out-of-range slider values are clamped. */
export function customPackEntitlements(
  hoursPerDay: number,
  bidsPerMonth: number,
  opts?: { priority?: boolean; b2b?: boolean },
): PackEntitlements {
  const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(Math.round(v), lo), hi);
  const h = clamp(hoursPerDay, CUSTOM_PACK.hoursMin, CUSTOM_PACK.hoursMax);
  const b = clamp(bidsPerMonth, CUSTOM_PACK.bidsMin, CUSTOM_PACK.bidsMax);
  const priority = opts?.priority === true;
  const b2b      = opts?.b2b === true;
  return {
    price: CUSTOM_PACK.basePrice
      + (h - CUSTOM_PACK.hoursMin) * CUSTOM_PACK.hourPrice
      + b * CUSTOM_PACK.bidPrice
      + (priority ? CUSTOM_PACK.priorityPrice : 0)
      + (b2b ? CUSTOM_PACK.b2bPrice : 0),
    dailyQuotaSeconds: h * 3600,
    monthlyBidQuota: b,
    searchPriority: priority,
    b2bAccess: b2b,
  };
}

// Africa/Algiers = UTC+1, no DST.
// ponytail: fixed +1h offset instead of a tz library — Algeria has no DST;
// switch to Intl/luxon only if the app ever spans timezones.
const ALGIERS_OFFSET_MS = 60 * 60 * 1000;

/** Local day key YYYY-MM-DD for the daily usage bucket. */
export function algiersDayKey(d: Date): string {
  return new Date(d.getTime() + ALGIERS_OFFSET_MS).toISOString().slice(0, 10);
}

/** Local month key YYYY-MM for the monthly bid bucket. */
export function algiersMonthKey(d: Date): string {
  return algiersDayKey(d).slice(0, 7);
}

/** Seconds elapsed since local midnight — caps a session credit at day start. */
export function secondsSinceAlgiersMidnight(d: Date): number {
  const local = d.getTime() + ALGIERS_OFFSET_MS;
  return Math.floor((local % (24 * 60 * 60 * 1000)) / 1000);
}

/**
 * Mongo filter enforcing the full visibility contract for worker discovery
 * (search + map). A worker is discoverable only when BOTH hold:
 *   1. subscription active and not expired;
 *   2. daily quota not exhausted — today's accumulated bucket PLUS the live
 *      session (clipped at local midnight) stays under the doc's own
 *      dailyQuotaSeconds. Unlimited packs (null) always pass.
 * All packs are 7/7 — no calendar-day rule.
 *
 * Reads only STORED entitlement fields — presets and custom packs go through
 * the same math. Legacy docs without the fields pass ($ifNull grandfather).
 * Both discovery paths (UsersService.findMany subscribedOnly and
 * LocationService.getWorkersInCell) spread this in, so the rules can never
 * drift apart.
 */
export function subscriptionVisibilityFilter(now: Date): Record<string, unknown> {
  const filter: Record<string, unknown> = {
    subscriptionActive: true,
    subscriptionUntil:  { $gt: now },
  };

  // usedToday = (bucket if it belongs to today) + (live session if online,
  // started no earlier than local midnight, never negative on clock skew).
  const today    = algiersDayKey(now);
  const midnight = new Date(now.getTime() - secondsSinceAlgiersMidnight(now) * 1000);
  const usedToday = {
    $add: [
      { $cond: [{ $eq: ['$usageDay', today] }, { $ifNull: ['$usageSeconds', 0] }, 0] },
      {
        $cond: [
          { $and: [{ $eq: ['$isOnline', true] }, { $gt: ['$onlineSince', null] }] },
          {
            $max: [
              0,
              { $divide: [{ $subtract: [now, { $max: ['$onlineSince', midnight] }] }, 1000] },
            ],
          },
          0,
        ],
      },
    ],
  };

  filter.$expr = {
    $lt: [usedToday, { $ifNull: ['$dailyQuotaSeconds', Number.MAX_SAFE_INTEGER] }],
  };
  return filter;
}

export type UserDocument = User & Document;

/**
 * Backward-compatible type alias so existing code importing WorkerDocument
 * from this module compiles without changes.
 */
export type WorkerDocument = UserDocument;

@Schema({ collection: 'users', timestamps: false, versionKey: false })
export class User {
  // ── Identity ────────────────────────────────────────────────────────────────
  @Prop({ required: true })
  _id: string;                         // Firebase UID — same for client & worker
  // NOTE: no `index: true` — MongoDB always indexes _id. Declaring it again
  // triggers "Can not overwrite the default _id index" on startup.

  @Prop({ required: true })
  name: string;

  /**
   * Email — optionnel pour les utilisateurs Phone Auth (peut être '').
   * L'index MongoDB est partiel : unicité appliquée uniquement si email ≠ ''.
   */
  // No lowercase/trim setters: the value is stored AES-256-GCM-encrypted, and a
  // string setter would mangle the base64 ciphertext. Normalization (trim +
  // lowercase) for matching/uniqueness happens inside blindIndex().
  @Prop({ default: '' })
  email: string;

  /**
   * Numéro de téléphone au format E.164 (+213XXXXXXXXX pour l'Algérie).
   * Champ principal d'identification pour les utilisateurs Phone Auth.
   * Index partiel : uniquement si phoneNumber ≠ ''.
   */
  @Prop({ default: '' })
  phoneNumber: string;

  // ── Blind indexes (populated by the encryptedFields plugin) ─────────────────
  // Deterministic HMAC of email/phoneNumber. The plaintext fields above are
  // stored AES-256-GCM-encrypted (non-searchable); these carry the uniqueness
  // constraint and enable exact-match lookup. Never set these by hand.
  @Prop({ default: '', select: false })
  emailBidx: string;

  @Prop({ default: '', select: false })
  phoneNumberBidx: string;

  @Prop({
    required: true,
    enum: Object.values(UserRole),
    default: UserRole.Client,
    index: true,
  })
  role: UserRole;

  // ── Moderation (admin-managed) ───────────────────────────────────────────────
  // Additive fields consumed by the admin dashboard. Defaults keep every legacy
  // document (and every app query) behaving exactly as before.

  /** Set by an admin to block a user from the platform. */
  @Prop({ default: false, index: true })
  isBanned: boolean;

  /** Set by an admin to mark a worker profile as verified/trusted. */
  @Prop({ default: false })
  isVerified: boolean;

  // ── Document verification / approval (worker optional, business mandatory) ────
  // Business logic (memoire): a company account MUST submit legal documents and
  // be approved by an admin before it can sign in. A worker MAY submit documents
  // (optional). Clients and doc-less workers stay 'approved' — the gate is a
  // no-op for them. The FirebaseAuthGuard blocks 'pending'/'rejected' the same
  // way the PIN gate blocks unknown devices, so an un-approved account cannot
  // touch any endpoint until an admin clears it.
  //
  //   approved → default; full access. Doc-less workers/clients never leave it.
  //   pending  → docs submitted, awaiting admin review; every request 403s.
  //   rejected → admin declined; the app shows the note and offers resubmit.

  /** '' (approved) | 'pending' | 'rejected'. Empty = approved (default). */
  @Prop({ type: String, default: '', index: true })
  verificationStatus: string;

  /** Cloudinary URLs of submitted legal/identity documents (PDF or image). */
  @Prop({ type: [String], default: [] })
  verificationDocs: string[];

  /** Admin's note on rejection (shown to the user so they can fix + resubmit). */
  @Prop({ type: String, default: '' })
  verificationNote: string;

  // ── Location (shared) ────────────────────────────────────────────────────────
  @Prop({ type: Number, default: null })
  latitude: number | null;

  @Prop({ type: Number, default: null })
  longitude: number | null;

  @Prop({ required: true, type: Date })
  lastUpdated: Date;

  @Prop({ type: String, default: null })
  cellId: string | null;

  @Prop({ type: Number, default: null })
  wilayaCode: number | null;

  @Prop({ type: String, default: null })
  geoHash: string | null;

  @Prop({ type: Date, default: null })
  lastCellUpdate: Date | null;

  /**
   * UI language ('fr' | 'ar' | 'en'), sent by the client on profile create/
   * update. Used to render server-side notifications (push + inbox) in the
   * recipient's language. Defaults to 'fr'.
   */
  @Prop({ type: String, default: 'fr' })
  language: string;

  // ── Account PIN (anti SIM-recycling) ─────────────────────────────────────────
  // Optional 6-digit PIN (WhatsApp two-step model): recycled Algerian SIMs let a
  // buyer pass phone-OTP and inherit the account. When pinHash is set, requests
  // from a device not in knownDevices are rejected (403 PIN_REQUIRED) until
  // POST /auth/verify-pin succeeds. All fields select:false — they must never
  // ride along on user documents returned by the API.

  /** scrypt hash (`s1:salt:hash`, see common/crypto/pin-hash.ts). '' = no PIN. */
  @Prop({ default: '', select: false })
  pinHash: string;

  /** Client-generated random device ids that have passed PIN verification. */
  @Prop({ type: [String], default: [], select: false })
  knownDevices: string[];

  /** Consecutive failed PIN attempts — reset on success. */
  @Prop({ default: 0, select: false })
  pinFailedAttempts: number;

  /** Lockout expiry after too many failed attempts. null = not locked. */
  @Prop({ type: Date, default: null, select: false })
  pinLockedUntil: Date | null;

  /**
   * Forgotten-PIN reset request timestamp (WhatsApp model): 7 days after this,
   * the next verify attempt clears the PIN. Cancelled by any successful verify.
   */
  @Prop({ type: Date, default: null, select: false })
  pinResetRequestedAt: Date | null;

  // ── Media / push (shared) ────────────────────────────────────────────────────
  @Prop({ type: String, default: null })
  profileImageUrl: string | null;

  @Prop({ type: String, default: null })
  fcmToken: string | null;

  /**
   * Notification `type` keys the user muted in Settings (e.g. 'bid_received',
   * 'admin_broadcast'). Gates the FCM push server-side — the inbox entry is
   * still written. Synced from the client's Settings toggles; the default
   * mirrors the client default (promotions are opt-in, everything else on).
   */
  @Prop({ type: [String], default: ['admin_broadcast', 'promotion'] })
  mutedNotifTypes: string[];

  /**
   * Opt-in consent to log this user's AI queries (text/voice/photo) as
   * training data for the P6 flywheel (ai_query_logs collection). Asked once
   * in-app at first AI-search use, toggleable in Settings. Default OFF —
   * nothing is ever logged without it.
   */
  @Prop({ default: false })
  aiDataConsent: boolean;

  // ── Worker-specific ──────────────────────────────────────────────────────────
  // Defaults guarantee that client documents never satisfy worker-targeted
  // queries (e.g. { role: 'worker', isOnline: true }).

  /** Trade / profession key (null for clients). */
  @Prop({ type: String, default: null })
  profession: string | null;

  /**
   * Free-text "about me" shown on the public profile, worker-written. Public
   * and unformatted (Flutter renders it as plain Text — no markup path), length
   * capped by the DTO. '' = nothing written, and the section is not rendered.
   */
  @Prop({ type: String, default: '', maxlength: 300 })
  bio: string;

  /**
   * Portfolio — Cloudinary image URLs of the worker's past jobs, newest last,
   * client-ordered. Public: it ships with the worker document to search, map
   * preview and the public profile. Writes go through the dedicated endpoint
   * (PUT /workers/:id/portfolio), never through the generic profile upsert:
   * the length is capped by portfolioQuota and every URL must be an image this
   * worker uploaded to our own Cloudinary account.
   */
  @Prop({ type: [String], default: [] })
  portfolio: string[];

  /** Online status — meaningful only for workers. Always false for clients. */
  @Prop({ default: false })
  isOnline: boolean;

  /** Bayesian average rating (0–5). */
  @Prop({ default: 0.0, min: 0, max: 5 })
  averageRating: number;

  @Prop({ default: 0, min: 0 })
  ratingCount: number;

  /** Running sum of stars — enables Bayesian recomputation without history. */
  @Prop({ default: 0, min: 0 })
  ratingSum: number;

  @Prop({ default: 0, min: 0 })
  jobsCompleted: number;

  /** Fraction of bids responded to (0–1). */
  @Prop({ default: 0.7, min: 0, max: 1 })
  responseRate: number;

  /** Timestamp of last offline transition — used for recency ranking. */
  @Prop({ type: Date, default: null })
  lastActiveAt: Date | null;

  // ── Subscription (worker visibility) ─────────────────────────────────────────
  // The business model (memoire): workers pay a visibility subscription — no
  // commission. An unsubscribed worker keeps full client access but is hidden
  // from search/map and cannot bid. Defaults keep every client document inert.

  /** Worker visibility subscription active. Gates search visibility + bidding. */
  @Prop({ default: false, index: true })
  subscriptionActive: boolean;

  /** Subscription expiry. null = never subscribed. */
  @Prop({ type: Date, default: null })
  subscriptionUntil: Date | null;

  /**
   * Subscription pack id: basic | pro | business | expert | custom.
   * The gates never read this — they read the stored entitlement fields
   * below, written together at activation. null = never subscribed.
   */
  @Prop({ type: String, enum: [...SUBSCRIPTION_TIERS, null], default: null })
  subscriptionTier: SubscriptionTier | null;

  // ── Stored entitlements (written at activation, read by the gates) ──────────

  /** Pack price in DZD/month (informational — shown in the app). */
  @Prop({ type: Number, default: null })
  subscriptionPrice: number | null;

  /** Daily map-visibility quota in seconds. null = unlimited. */
  @Prop({ type: Number, default: null })
  dailyQuotaSeconds: number | null;

  /** Monthly bid quota. null = unlimited, 0 = no bid access (Basic). */
  @Prop({ type: Number, default: null })
  monthlyBidQuota: number | null;

  /** Business/Expert: ranking boost in discovery + "Pro" badge. */
  @Prop({ default: false })
  searchPriority: boolean;

  /**
   * Number of portfolio photos the pack allows (see portfolioQuotaForPrice).
   * null = never subscribed → no portfolio. Legacy docs that subscribed before
   * this field existed are grandfathered from subscriptionPrice at write time.
   */
  @Prop({ type: Number, default: null })
  portfolioQuota: number | null;

  /**
   * Expert-tier ("accès flux B2B"). When true and the subscription is active,
   * the worker also surfaces in Business-account discovery. Additive: it never
   * hides the worker from normal client search. Kept as a flag (denormalized
   * from the pack) because the b2bOnly search filter indexes on it.
   */
  @Prop({ default: false })
  b2bAccess: boolean;

  // ── Bid metering (monthly bucket, mirrors the daily usage bucket) ───────────

  /** Bids consumed in [bidMonth]. */
  @Prop({ default: 0, min: 0 })
  bidsUsed: number;

  /** Local month (YYYY-MM, Africa/Algiers) the bidsUsed belong to. */
  @Prop({ type: String, default: null })
  bidMonth: string | null;

  // ── Usage metering (worker online time, per-day) ─────────────────────────────
  // The worker's paid visibility is metered in online hours PER DAY — the
  // counter resets at local midnight (usageDay tracks which day the seconds
  // belong to). We accumulate the elapsed online duration each time the worker
  // goes offline; the live counter (usageSeconds + now−onlineSince) is computed
  // client-side while online, with the same day-rollover rule.

  /** Online time accumulated for `usageDay`, in seconds. */
  @Prop({ default: 0, min: 0 })
  usageSeconds: number;

  /** Local day (YYYY-MM-DD, Africa/Algiers) the usageSeconds belong to. */
  @Prop({ type: String, default: null })
  usageDay: string | null;

  /** Instant the worker last went online. null when offline. */
  @Prop({ type: Date, default: null })
  onlineSince: Date | null;
}

export const UserSchema = SchemaFactory.createForClass(User);

// PII field encryption (AES-256-GCM) + blind index. Must be applied before the
// blind-index unique indexes below are used at runtime.
UserSchema.plugin(encryptedFields, { fields: [...USER_ENCRYPTED_FIELDS] });

// ── Shared indexes ────────────────────────────────────────────────────────────

/**
 * Uniqueness now lives on the BLIND INDEX, not the plaintext field: `email` and
 * `phoneNumber` hold non-deterministic AES-256-GCM ciphertext, so a unique index
 * on them would be meaningless. `emailBidx`/`phoneNumberBidx` are deterministic
 * HMACs, so the same email/phone always yields the same index → real uniqueness.
 *
 * ⚠️ MIGRATION (run once, in order):
 *   1. Backfill: `node dist/scripts/encrypt-existing-pii.js`  (encrypts + fills bidx)
 *   2. db.users.dropIndex("email_1"); db.users.dropIndex("phoneNumber_1");
 *   3. Deploy this version (autoIndex builds the bidx unique indexes in dev;
 *      in prod run the createIndex calls below manually).
 */
UserSchema.index(
  { emailBidx: 1 },
  { unique: true, partialFilterExpression: { emailBidx: { $ne: '' } } },
);

UserSchema.index(
  { phoneNumberBidx: 1 },
  { unique: true, partialFilterExpression: { phoneNumberBidx: { $ne: '' } } },
);

UserSchema.index({ wilayaCode: 1 });
UserSchema.index({ geoHash: 1 });

// ── Partial indexes (role = 'worker' documents only) ──────────────────────────
// MongoDB partial indexes only maintain index entries for documents satisfying
// the partialFilterExpression. Client documents are invisible to these indexes,
// keeping storage and write amplification proportional to the worker count.
const WORKER_ONLY = { partialFilterExpression: { role: UserRole.Worker } } as const;

UserSchema.index({ isOnline: 1, wilayaCode: 1 },             WORKER_ONLY);
UserSchema.index({ isOnline: 1, profession: 1 },             WORKER_ONLY);
UserSchema.index({ wilayaCode: 1, profession: 1 },           WORKER_ONLY);
UserSchema.index({ cellId: 1, profession: 1, isOnline: 1 },  WORKER_ONLY);
UserSchema.index({ wilayaCode: 1, isOnline: 1 },             WORKER_ONLY);
