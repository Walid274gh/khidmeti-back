// ══════════════════════════════════════════════════════════════════════════════
// PinGateService — the single enforcement point for the optional account PIN.
//
// THREAT MODEL (Algeria SIM recycling):
//   Inactive numbers are resold. The buyer receives the OTP and passes Firebase
//   Phone Auth with the ORIGINAL owner's uid. Phone possession alone therefore
//   cannot be trusted for accounts that opted into a PIN. The PIN is a secret
//   the SIM buyer does not have.
//
// ENFORCEMENT (server-side, not a client screen):
//   Every HTTP request (FirebaseAuthGuard) and WS handshake (3 gateways) calls
//   isDeviceAllowed(uid, deviceId). If the account has a PIN and the device id
//   is not in knownDevices, the request is rejected with PIN_REQUIRED until
//   POST /auth/verify-pin succeeds from that device. Omitting the X-Device-Id
//   header is rejected the same way — a header the attacker can drop is not a
//   gate, so absence of the header counts as an unknown device.
//
// BRUTE FORCE: a 6-digit PIN has 10^6 combinations, so the hash alone is not
// the defense — the attempt counter is. 5 consecutive failures lock PIN
// verification for 15 minutes (server-side, per account).
//
// RECOVERY (WhatsApp model): a user who forgot their PIN calls
// POST /auth/request-pin-reset; after a 7-day cooling period the PIN clears
// on their next verify attempt. No SMS/email recovery — SMS is exactly the
// channel this feature distrusts. The 7 days give the real owner (who gets
// pushed a notification and still has a logged-in device) time to cancel.
// ══════════════════════════════════════════════════════════════════════════════

import { Injectable, Logger, NotFoundException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { User, UserDocument } from '../../schemas/user.schema';
import { hashPin, verifyPin } from '../../common/crypto/pin-hash';

const PIN_FIELDS =
  'pinHash knownDevices pinFailedAttempts pinLockedUntil pinResetRequestedAt verificationStatus';

const MAX_ATTEMPTS   = 5;
const LOCK_MS        = 15 * 60 * 1000;
const RESET_COOL_MS  = 7 * 24 * 60 * 60 * 1000;
const MAX_DEVICES    = 10;
// ponytail: in-memory cache, single api container (same tradeoff as the
// ThrottlerGuard store). Multi-replica → move to ioredis (already a dep).
const CACHE_TTL_MS   = 30 * 1000;

interface PinDoc {
  pinHash?: string;
  knownDevices?: string[];
  pinFailedAttempts?: number;
  pinLockedUntil?: Date | null;
  pinResetRequestedAt?: Date | null;
  verificationStatus?: string;
}

export type VerifyPinResult =
  | { ok: true }
  | { ok: false; reason: 'wrong_pin' | 'locked' | 'no_pin'; attemptsLeft?: number };

@Injectable()
export class PinGateService {
  private readonly logger = new Logger(PinGateService.name);
  private readonly cache = new Map<string, { doc: PinDoc | null; at: number }>();

  constructor(
    @InjectModel(User.name)
    private readonly userModel: Model<UserDocument>,
  ) {}

  // ── Gate check (hot path — every request / WS handshake) ────────────────────

  /**
   * True when the request may proceed: account has no PIN, or the device id is
   * already verified. Fails OPEN on DB errors — a Mongo blip must not lock the
   * whole user base out; the attacker still has to beat the normal path.
   */
  async isDeviceAllowed(uid: string, deviceId: string | undefined): Promise<boolean> {
    try {
      const doc = await this.load(uid);
      if (!doc?.pinHash) return true;               // no PIN set (or no profile yet)
      if (!deviceId) return false;                  // PIN set + anonymous device → gate
      return (doc.knownDevices ?? []).includes(deviceId);
    } catch (err) {
      this.logger.warn(`isDeviceAllowed(${uid}) failed open: ${(err as Error).message}`);
      return true;
    }
  }

  /** For /auth/check: does this account have a PIN, and does THIS device pass? */
  async status(uid: string, deviceId: string | undefined): Promise<{ hasPin: boolean; pinRequired: boolean }> {
    const doc = await this.load(uid);
    const hasPin = !!doc?.pinHash;
    return {
      hasPin,
      pinRequired: hasPin && (!deviceId || !(doc?.knownDevices ?? []).includes(deviceId)),
    };
  }

  // ── Document-approval gate (worker optional, business mandatory) ──────────────
  // Reuses the same cached load() as the PIN gate — no extra per-request read.
  // An account whose verificationStatus is 'pending' or 'rejected' has submitted
  // documents and is awaiting (or was denied) admin approval; it must be blocked
  // from every endpoint just like an un-verified device. '' (approved) and any
  // account with no profile yet pass — the setup POST itself must go through.

  /** True when the account may proceed (approved, or no docs submitted). */
  async isApproved(uid: string): Promise<boolean> {
    try {
      const doc = await this.load(uid);
      const status = doc?.verificationStatus ?? '';
      return status !== 'pending' && status !== 'rejected';
    } catch (err) {
      // Fail OPEN, same rationale as isDeviceAllowed: a DB blip must not lock
      // the whole user base out.
      this.logger.warn(`isApproved(${uid}) failed open: ${(err as Error).message}`);
      return true;
    }
  }

  /** Drop the cached doc so a fresh approval status is read on the next request. */
  invalidate(uid: string): void {
    this.cache.delete(uid);
  }

  // ── PIN lifecycle ────────────────────────────────────────────────────────────

  /**
   * Set (or change) the PIN and trust ONLY the calling device (WhatsApp
   * model: rotating the PIN untrusts every other device — each re-verifies
   * with the new PIN; the app's PIN_REQUIRED hook lands them on the PIN
   * screen, not on dead errors).
   */
  async setPin(uid: string, pin: string, deviceId: string): Promise<void> {
    const res = await this.userModel.updateOne(
      { _id: uid },
      {
        $set: {
          pinHash:             hashPin(pin),
          knownDevices:        [deviceId],
          pinFailedAttempts:   0,
          pinLockedUntil:      null,
          pinResetRequestedAt: null,
        },
      },
    ).exec();
    // No profile document = nothing was protected; a silent "saved" here
    // would be a lie the user pays for at the next sign-in.
    if (res.matchedCount === 0) {
      throw new NotFoundException('No profile for this account');
    }
    this.cache.delete(uid);
    this.logger.log(`PIN set for uid=${uid}`);
  }

  /** Remove the PIN entirely (requires current PIN — enforced by controller). */
  async removePin(uid: string): Promise<void> {
    await this.userModel.updateOne(
      { _id: uid },
      {
        $set: {
          pinHash:             '',
          knownDevices:        [],
          pinFailedAttempts:   0,
          pinLockedUntil:      null,
          pinResetRequestedAt: null,
        },
      },
    ).exec();
    this.cache.delete(uid);
    this.logger.log(`PIN removed for uid=${uid}`);
  }

  /**
   * Verify the PIN from [deviceId]. On success the device joins knownDevices.
   * Handles: lockout window, attempt counting, and the 7-day forgotten-PIN
   * reset (if the cooling period elapsed, the PIN is cleared and verification
   * succeeds trivially — the account is back to OTP-only).
   */
  async verify(uid: string, pin: string, deviceId: string): Promise<VerifyPinResult> {
    const doc = await this.load(uid, /*fresh*/ true);
    if (!doc?.pinHash) return { ok: false, reason: 'no_pin' };

    // Forgotten-PIN reset window elapsed → clear the PIN, let the user in.
    if (doc.pinResetRequestedAt &&
        Date.now() - new Date(doc.pinResetRequestedAt).getTime() >= RESET_COOL_MS) {
      await this.removePin(uid);
      this.logger.warn(`PIN auto-cleared after 7-day reset window uid=${uid}`);
      return { ok: true };
    }

    if (doc.pinLockedUntil && new Date(doc.pinLockedUntil).getTime() > Date.now()) {
      return { ok: false, reason: 'locked' };
    }

    if (!verifyPin(pin, doc.pinHash)) {
      const attempts = (doc.pinFailedAttempts ?? 0) + 1;
      const update: Record<string, unknown> = { pinFailedAttempts: attempts };
      if (attempts >= MAX_ATTEMPTS) {
        update['pinFailedAttempts'] = 0;
        update['pinLockedUntil']    = new Date(Date.now() + LOCK_MS);
        this.logger.warn(`PIN locked for uid=${uid} (${MAX_ATTEMPTS} failures)`);
      }
      await this.userModel.updateOne({ _id: uid }, { $set: update }).exec();
      this.cache.delete(uid);
      return attempts >= MAX_ATTEMPTS
        ? { ok: false, reason: 'locked' }
        : { ok: false, reason: 'wrong_pin', attemptsLeft: MAX_ATTEMPTS - attempts };
    }

    // Success — trust this device, reset counters, cancel any pending reset.
    // Skip the push when already trusted (change/remove flows re-verify from a
    // known device): $push has no dedup, and duplicates would eat the
    // MAX_DEVICES slots and evict genuine older devices via $slice.
    const alreadyKnown = (doc.knownDevices ?? []).includes(deviceId);
    await this.userModel.updateOne(
      { _id: uid },
      {
        ...(alreadyKnown
          ? {}
          : { $push: { knownDevices: { $each: [deviceId], $slice: -MAX_DEVICES } } }),
        $set: { pinFailedAttempts: 0, pinLockedUntil: null, pinResetRequestedAt: null },
      },
    ).exec();
    this.cache.delete(uid);
    this.logger.log(`PIN verified, device trusted uid=${uid}`);
    return { ok: true };
  }

  /** Start the 7-day forgotten-PIN cooling period. Idempotent. */
  // ponytail: no owner push notification yet — PushSenderService lives in
  // NotificationsModule which imports AuthModule (circular). Add via
  // forwardRef when the reset flow ships to real users.
  async requestReset(uid: string): Promise<Date> {
    const doc = await this.load(uid, /*fresh*/ true);
    const existing = doc?.pinResetRequestedAt ? new Date(doc.pinResetRequestedAt) : null;
    if (existing) return new Date(existing.getTime() + RESET_COOL_MS);

    const now = new Date();
    await this.userModel
      .updateOne({ _id: uid }, { $set: { pinResetRequestedAt: now } })
      .exec();
    this.cache.delete(uid);
    this.logger.warn(`PIN reset requested uid=${uid} — clears in 7 days`);
    return new Date(now.getTime() + RESET_COOL_MS);
  }

  // ── Internals ────────────────────────────────────────────────────────────────

  private async load(uid: string, fresh = false): Promise<PinDoc | null> {
    if (!fresh) {
      const hit = this.cache.get(uid);
      if (hit && Date.now() - hit.at < CACHE_TTL_MS) return hit.doc;
    }
    const doc = await this.userModel
      .findById(uid)
      .select(PIN_FIELDS)
      .lean()
      .exec() as PinDoc | null;
    this.cache.set(uid, { doc, at: Date.now() });
    return doc;
  }
}
