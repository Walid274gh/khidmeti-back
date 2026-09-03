// apps/api/src/modules/users/users.service.ts
//
// ADDED: ensureExists(uid, claims)
//
// ROOT CAUSE of the "Erreur lors de l'envoi" bug (Image 6):
//   A Firebase Auth account existed (JWT was valid → FirebaseAuthGuard passed),
//   but the corresponding MongoDB document in the 'users' collection had never
//   been created.  This happens when:
//     1. The Flutter app created the Firebase account successfully, AND
//     2. The network call to POST /users (createOrUpdateUser) failed silently
//        during registration (timeout, app killed, no network), OR
//     3. _ensureBackendProfile() in AuthService fired after signIn but the
//        request never reached the server.
//
// FIX STRATEGY — "upsert on demand":
//   UsersController.findById() calls ensureExists() when the requester is
//   querying their own uid.  ensureExists() is idempotent: if the document
//   already exists it is returned unchanged.  If not, a minimal 'client' profile
//   is created from the Firebase token claims (email, displayName).  This is
//   safe because:
//     • The JWT has already been verified by FirebaseAuthGuard.
//     • We only auto-provision for the authenticated user's OWN uid.
//     • The created document has role='client' — correct for a new user.
//     • A subsequent POST /users from the Flutter app will upsert additional
//       fields (phone, profileImageUrl, etc.) over the auto-provisioned stub.

import {
  BadRequestException,
  ForbiddenException,
  Injectable,
  Logger,
  NotFoundException,
} from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { FilterQuery, Model } from 'mongoose';
import {
  User,
  UserDocument,
  UserRole,
  SubscriptionTier,
  PackEntitlements,
  TIER_PACKS,
  CUSTOM_PACK,
  customPackEntitlements,
  portfolioQuotaForPrice,
  PORTFOLIO_PREVIEW_COUNT,
  algiersDayKey,
  algiersMonthKey,
  secondsSinceAlgiersMidnight,
  subscriptionVisibilityFilter,
  EXTRA_BIDS_PACK_SIZE,
  ANNUAL_MULTIPLIER,
  MONTHLY_DAYS,
  ANNUAL_DAYS,
} from '../../schemas/user.schema';
import { PushSenderService } from '../notifications/push-sender.service';
import { VectorIndexService } from '../../qdrant/vector-index.service';
import { CreateUserDto }    from '../../dto/create-user.dto';
import { UpdateUserDto }    from '../../dto/update-user.dto';
import { CreateWorkerDto }  from '../../dto/create-worker.dto';
import { UpdateWorkerDto }  from '../../dto/update-worker.dto';

// ── Filter shapes ─────────────────────────────────────────────────────────────

export interface UserFilters {
  role?: UserRole;
  wilayaCode?: number;
  profession?: string;
  isOnline?: boolean;
  cellId?: string;
  /** Restrict to these uids — semantic search hydrates Qdrant hits through
   *  this so vector results obey the exact same visibility rules. */
  ids?: string[];
  limit?: number;
  /** When true, only workers with an active, unexpired visibility subscription. */
  subscribedOnly?: boolean;
  /** When true, only Expert-tier (b2bAccess) workers — the Business-account view. */
  b2bOnly?: boolean;
}

export interface ProvisionClaims {
  /** Firebase displayName — used as the account's name. */
  name?: string | undefined;
  /** Firebase email — used as the account's email. */
  email?: string | undefined;
}

// ──────────────────────────────────────────────────────────────────────────────

@Injectable()
export class UsersService {
  private readonly logger = new Logger(UsersService.name);

  constructor(
    @InjectModel(User.name)
    private readonly userModel: Model<UserDocument>,
    private readonly pushSender: PushSenderService,
    private readonly vectorIndex: VectorIndexService,
  ) {}

  // ═══════════════════════════════════════════════════════════════════════════
  // SHARED (client + worker)
  // ═══════════════════════════════════════════════════════════════════════════

  async upsert(dto: CreateUserDto | CreateWorkerDto): Promise<UserDocument> {
    try {
      const role = dto.role ?? UserRole.Client;
      const patch: Partial<Record<string, unknown>> = {
        name:        dto.name,
        email:       dto.email,
        role,
        phoneNumber: dto.phoneNumber ?? '',
        latitude:    dto.latitude    ?? null,
        longitude:   dto.longitude   ?? null,
        profileImageUrl: dto.profileImageUrl ?? null,
        fcmToken:    dto.fcmToken    ?? null,
        lastUpdated: new Date(),
      };

      if (dto.language)                                 patch['language']   = dto.language;
      if ('profession' in dto && dto.profession)       patch['profession'] = dto.profession;
      if ('isOnline'   in dto && dto.isOnline != null) patch['isOnline']   = dto.isOnline;
      // '' clears the bio — only an absent field means "leave it alone".
      if ('bio'        in dto && dto.bio      != null) patch['bio']        = dto.bio;

      // ── Document verification / approval ──────────────────────────────────
      // Business MUST submit documents; a worker MAY. When docs are present the
      // account enters admin review ('pending') and the auth gate blocks it
      // until an admin approves. Clients / doc-less workers stay 'approved'.
      const docs = ('verificationDocs' in dto ? dto.verificationDocs : undefined) ?? [];
      if (role === UserRole.Business && docs.length === 0) {
        // Only enforce at first submission — an approved business updating its
        // profile later shouldn't have to re-send its documents every time.
        const hasDocs = await this.userModel
          .exists({ _id: dto.id, 'verificationDocs.0': { $exists: true } })
          .exec();
        if (!hasDocs) {
          throw new BadRequestException('Business accounts must submit at least one document');
        }
      }
      if (docs.length > 0) {
        patch['verificationDocs']   = docs;
        patch['verificationStatus'] = 'pending';
        patch['verificationNote']   = '';
      }

      const doc = await this.userModel
        .findByIdAndUpdate(dto.id, patch, { upsert: true, new: true, runValidators: true })
        .exec();

      if (!doc) throw new NotFoundException(`User ${dto.id} not found after upsert`);

      // Keep the semantic-matching index in step with the profile write.
      // Fire-and-forget: Qdrant/embeddings down must never fail a submit.
      if (doc.role === UserRole.Worker) this.vectorIndex.indexWorker(doc);

      // Alert admins that a new document set awaits review (fire-and-forget).
      if (docs.length > 0) void this.notifyAdminsOfSubmission(dto.name);

      return doc;
    } catch (err) {
      this.logger.error('UsersService.upsert failed', err);
      throw err;
    }
  }

  /**
   * Push a "new documents to review" notification to every admin. Best-effort:
   * never throws (pushSender.notify swallows its own errors), so a notification
   * hiccup can't fail the account submission itself.
   */
  private async notifyAdminsOfSubmission(name: string): Promise<void> {
    try {
      const admins = await this.userModel
        .find({ role: UserRole.Admin })
        .select('_id')
        .lean()
        .exec();
      for (const a of admins as Array<{ _id: string }>) {
        void this.pushSender.notify(a._id, {
          type: 'verification_submitted',
          params: { name },
        });
      }
    } catch (err) {
      this.logger.warn(`notifyAdminsOfSubmission failed: ${(err as Error).message}`);
    }
  }

  async findById(id: string): Promise<UserDocument> {
    try {
      const doc = await this.userModel.findById(id).exec();
      if (!doc) throw new NotFoundException(`User ${id} not found`);
      return doc;
    } catch (err) {
      this.logger.error(`UsersService.findById(${id}) failed`, err);
      throw err;
    }
  }

  async findByIdOrNull(id: string): Promise<UserDocument | null> {
    return this.userModel.findById(id).exec();
  }

  /**
   * Cheap role lookup for request-time authorization branches (e.g. deciding
   * whether a search viewer is a Business account). `.lean()` + projection skips
   * document hydration and the PII-decryption plugin — no wasted crypto work.
   */
  async getRole(id: string): Promise<UserRole | null> {
    const doc = await this.userModel.findById(id).select('role').lean().exec();
    return (doc?.role as UserRole) ?? null;
  }

  /**
   * Idempotent "upsert on demand".
   *
   * Returns the existing document unchanged if it already exists.
   * If the user is absent from MongoDB (e.g. profile creation failed during
   * registration), creates a minimal stub from the Firebase token claims.
   *
   * This is the canonical fix for the "Erreur lors de l'envoi" 404 scenario:
   * the JWT is valid (Firebase Auth has the account), but the MongoDB profile
   * was never persisted.
   *
   * @param uid    Firebase UID — used as the document _id.
   * @param claims Decoded token fields (email, displayName) for the stub.
   */
  async ensureExists(uid: string, claims: ProvisionClaims): Promise<UserDocument> {
    // Fast path: document already exists — no write needed.
    const existing = await this.findByIdOrNull(uid);
    if (existing) return existing;

    // Derive a sensible display name from what the token gives us.
    const name =
      claims.name?.trim() ||
      claims.email?.split('@')[0] ||
      'User';

    this.logger.warn(
      `Auto-provisioning missing MongoDB profile for uid=${uid} ` +
      `(email=${claims.email ?? 'unknown'}) — this indicates a registration ` +
      `race condition.  The Flutter app will upsert the full profile shortly.`,
    );

    return this.upsert({
      id:    uid,
      name,
      email: claims.email ?? '',
      role:  UserRole.Client,
      // All other fields take their schema defaults:
      //   phoneNumber: '', latitude: null, longitude: null, etc.
    });
  }

  async findMany(filters: UserFilters): Promise<UserDocument[]> {
    try {
      const query: FilterQuery<User> = {};
      if (filters.role       != null) query.role       = filters.role;
      if (filters.wilayaCode != null) query.wilayaCode = filters.wilayaCode;
      if (filters.profession)         query.profession  = filters.profession;
      if (filters.isOnline   != null) query.isOnline   = filters.isOnline;
      if (filters.cellId)             query.cellId      = filters.cellId;
      if (filters.ids?.length)        query._id         = { $in: filters.ids };
      if (filters.subscribedOnly) {
        // Full visibility contract: active + not expired + daily quota not
        // exhausted. All packs are 7/7 — no calendar-day rule.
        Object.assign(query, subscriptionVisibilityFilter(new Date()));
      }
      if (filters.b2bOnly) query.b2bAccess = true;

      const limit = Math.min(filters.limit ?? 100, 200);
      // Business/Expert priority: they fill the limited result window first.
      // portfolio is $slice'd to the preview count — a list response has no use
      // for full galleries (see PORTFOLIO_PREVIEW_COUNT). $slice on its own is
      // not an inclusive projection, so every other field still comes back.
      return this.userModel
        .find(query)
        .slice('portfolio', PORTFOLIO_PREVIEW_COUNT)
        .sort({ searchPriority: -1 })
        .limit(limit)
        .exec();
    } catch (err) {
      this.logger.error('UsersService.findMany failed', err);
      throw err;
    }
  }

  async update(id: string, dto: UpdateUserDto): Promise<UserDocument> {
    try {
      const patch: Partial<Record<string, unknown>> = { lastUpdated: new Date() };
      if (dto.name             != null) patch['name']            = dto.name;
      if (dto.phoneNumber      != null) patch['phoneNumber']     = dto.phoneNumber;
      if (dto.profileImageUrl  != null) patch['profileImageUrl'] = dto.profileImageUrl;
      if (dto.cellId           != null) patch['cellId']          = dto.cellId;
      if (dto.wilayaCode       != null) patch['wilayaCode']      = dto.wilayaCode;
      if (dto.geoHash          != null) patch['geoHash']         = dto.geoHash;

      const doc = await this.userModel
        .findByIdAndUpdate(id, patch, { new: true, runValidators: true })
        .exec();
      if (!doc) throw new NotFoundException(`User ${id} not found`);
      return doc;
    } catch (err) {
      this.logger.error(`UsersService.update(${id}) failed`, err);
      throw err;
    }
  }

  async updateLocation(
    id: string,
    latitude: number,
    longitude: number,
    cellId?: string,
    wilayaCode?: number,
    geoHash?: string,
  ): Promise<void> {
    try {
      const patch: Partial<Record<string, unknown>> = {
        latitude,
        longitude,
        lastUpdated: new Date(),
      };
      if (cellId     != null) { patch['cellId']       = cellId;     patch['lastCellUpdate'] = new Date(); }
      if (wilayaCode != null)   patch['wilayaCode']   = wilayaCode;
      if (geoHash    != null)   patch['geoHash']      = geoHash;

      const result = await this.userModel.updateOne({ _id: id }, patch).exec();
      if (result.matchedCount === 0) throw new NotFoundException(`User ${id} not found`);
    } catch (err) {
      this.logger.error(`UsersService.updateLocation(${id}) failed`, err);
      throw err;
    }
  }

  async updateFcmToken(id: string, fcmToken: string): Promise<void> {
    try {
      const result = await this.userModel
        .updateOne({ _id: id }, { fcmToken, lastUpdated: new Date() })
        .exec();
      if (result.matchedCount === 0) throw new NotFoundException(`User ${id} not found`);
    } catch (err) {
      this.logger.error(`UsersService.updateFcmToken(${id}) failed`, err);
      throw err;
    }
  }

  async clearFcmToken(id: string): Promise<void> {
    await this.userModel.updateOne({ _id: id }, { fcmToken: null, lastUpdated: new Date() }).exec();
  }

  async updateNotificationPrefs(id: string, mutedNotifTypes: string[]): Promise<void> {
    try {
      const result = await this.userModel
        .updateOne({ _id: id }, { mutedNotifTypes, lastUpdated: new Date() })
        .exec();
      if (result.matchedCount === 0) throw new NotFoundException(`User ${id} not found`);
    } catch (err) {
      this.logger.error(`UsersService.updateNotificationPrefs(${id}) failed`, err);
      throw err;
    }
  }

  /** C1 flywheel — opt-in consent to log AI queries as training data. */
  async updateAiConsent(id: string, aiDataConsent: boolean): Promise<void> {
    try {
      const result = await this.userModel
        .updateOne({ _id: id }, { aiDataConsent, lastUpdated: new Date() })
        .exec();
      if (result.matchedCount === 0) throw new NotFoundException(`User ${id} not found`);
    } catch (err) {
      this.logger.error(`UsersService.updateAiConsent(${id}) failed`, err);
      throw err;
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // WORKER API  (all queries enforce role = 'worker')
  // ═══════════════════════════════════════════════════════════════════════════

  async upsertWorker(dto: CreateWorkerDto): Promise<UserDocument> {
    return this.upsert({ ...dto, role: UserRole.Worker });
  }

  async findWorkerById(id: string): Promise<UserDocument> {
    try {
      const doc = await this.userModel
        .findOne({ _id: id, role: UserRole.Worker })
        .exec();
      if (!doc) throw new NotFoundException(`Worker ${id} not found`);
      return doc;
    } catch (err) {
      this.logger.error(`UsersService.findWorkerById(${id}) failed`, err);
      throw err;
    }
  }

  async findWorkerByIdOrNull(id: string): Promise<UserDocument | null> {
    return this.userModel.findOne({ _id: id, role: UserRole.Worker }).exec();
  }

  async findWorkers(filters: Omit<UserFilters, 'role'>): Promise<UserDocument[]> {
    // Visibility gate: only subscribed workers are discoverable in search.
    return this.findMany({ ...filters, role: UserRole.Worker, subscribedOnly: true });
  }

  /**
   * Activate a worker's visibility subscription for `days` (default 30).
   *
   * Writes the pack's ENTITLEMENTS onto the doc (price, daily map quota,
   * monthly bid quota, priority, b2b) so the visibility and bid gates read
   * the doc alone — presets and slider-built custom packs are identical
   * downstream. Every field is set on every activation so a renewal on a
   * smaller pack always narrows the account. The monthly bid bucket resets
   * with each activation (new subscription month).
   *
   * B2B gate: expert preset and custom-with-B2B require ADMIN-VERIFIED docs
   * (isVerified — set by admin approval, admin.service). Throws Forbidden
   * 'DOCS_REQUIRED_FOR_B2B' otherwise; the app routes the worker to the
   * document submission flow.
   *
   * ponytail: stub payment — no gateway is called. Replace this body with a
   * SATIM (or other) payment-callback verification once merchant credentials
   * exist; keep the same field writes so the gates above keep working.
   */
  async activateSubscription(
    id: string,
    tier: SubscriptionTier = 'business',
    days = MONTHLY_DAYS,
    custom?: { hoursPerDay: number; bidsPerMonth: number; priority?: boolean; b2b?: boolean },
    period: 'monthly' | 'annual' = 'monthly',
  ): Promise<UserDocument> {
    try {
      if (tier === 'custom' && period === 'annual') {
        throw new BadRequestException('Annual billing is available for fixed tiers only');
      }
      const ent: PackEntitlements = tier === 'custom'
        ? customPackEntitlements(
            custom?.hoursPerDay ?? CUSTOM_PACK.hoursMin,
            custom?.bidsPerMonth ?? CUSTOM_PACK.bidsMin,
            { priority: custom?.priority, b2b: custom?.b2b },
          )
        : TIER_PACKS[tier];
      const billedDays = period === 'annual' ? ANNUAL_DAYS : days;
      const billedPrice = period === 'annual' && tier !== 'custom'
        ? TIER_PACKS[tier as Exclude<SubscriptionTier, 'custom'>].price * ANNUAL_MULTIPLIER
        : ent.price;

      if (ent.b2bAccess) {
        const w = await this.userModel
          .findById(id)
          .select('isVerified')
          .lean()
          .exec();
        if (!w) throw new NotFoundException(`User ${id} not found`);
        if ((w as { isVerified?: boolean }).isVerified !== true) {
          throw new ForbiddenException('DOCS_REQUIRED_FOR_B2B');
        }
      }

      const now = new Date();
      const until = new Date(now.getTime() + billedDays * 24 * 60 * 60 * 1000);

      // Extra rollover: if an extra pack was bought in the last 7 days of the
      // previous cycle, carry its remaining bids into the new cycle.
      const prev = await this.userModel.findById(id).select('extraBidsRemaining extraBidsExpiry extraBidsPurchasedAt subscriptionUntil').lean().exec() as { extraBidsRemaining?: number; extraBidsExpiry?: Date | null; extraBidsPurchasedAt?: Date | null; subscriptionUntil?: Date | null } | null;
      let carryExtra = 0;
      let carryExpiry: Date | null = null;
      let carryPurchasedAt: Date | null = null;
      if (prev && (prev.extraBidsRemaining ?? 0) > 0 && prev.extraBidsExpiry && prev.extraBidsPurchasedAt && prev.subscriptionUntil) {
        const msUntilExpiry = new Date(prev.subscriptionUntil).getTime() - new Date(prev.extraBidsPurchasedAt).getTime();
        const sevenDaysMs = 7 * 24 * 60 * 60 * 1000;
        if (msUntilExpiry <= sevenDaysMs && new Date(prev.extraBidsExpiry).getTime() > now.getTime()) {
          carryExtra = prev.extraBidsRemaining!;
          carryExpiry = prev.extraBidsExpiry as Date;
          carryPurchasedAt = prev.extraBidsPurchasedAt as Date;
        }
      }

      const doc = await this.userModel
        .findByIdAndUpdate(
          id,
          {
            subscriptionActive: true,
            subscriptionUntil:  until,
            subscriptionTier:   tier,
            subscriptionPrice:  billedPrice,
            subscriptionPeriod: period,
            subscriptionAnnual: period === 'annual',
            dailyQuotaSeconds:  ent.dailyQuotaSeconds,
            monthlyBidQuota:    ent.monthlyBidQuota,
            searchPriority:     ent.searchPriority,
            b2bAccess:          ent.b2bAccess,
            portfolioQuota:     portfolioQuotaForPrice(ent.price), // monthly-equivalent price: gallery size follows the tier, not the billing period
            bidsUsed:           0,
            bidMonth:           algiersMonthKey(now),
            extraBidsRemaining: carryExtra,
            extraBidsExpiry:    carryExpiry,
            extraBidsPurchasedAt: carryPurchasedAt,
            lastUpdated:        now,
          },
          { new: true },
        )
        .exec();
      if (!doc) throw new NotFoundException(`User ${id} not found`);
      return doc;
    } catch (err) {
      this.logger.error(`UsersService.activateSubscription(${id}) failed`, err);
      throw err;
    }
  }

  /**
   * Atomically consume one bid from the worker's monthly bucket.
   *
   * The bucket mirrors the daily usage bucket: `bidsUsed` counts within
   * `bidMonth` (Africa/Algiers), rolling over lazily on first bid of a new
   * month. The check-and-increment is a single guarded update — two devices
   * bidding concurrently can never overshoot the quota.
   *
   * Throws ForbiddenException with a machine-readable message the app maps:
   *   SUBSCRIPTION_REQUIRED — no active subscription;
   *   BID_NOT_INCLUDED      — pack has no bid access (Basic / custom 0 bids);
   *   BID_QUOTA_EXHAUSTED   — monthly quota used up.
   */
  /**
   * Purchase a 5-bid extra pack for 500 DA.
   * Rules: active subscription required, no active extra pack (extraBidsRemaining must be 0),
   * extra consumed first. Expiry = subscriptionUntil; if bought in last 7 days of cycle,
   * remaining extra rolls over on next activation.
   */
  async purchaseExtraBids(id: string): Promise<UserDocument> {
    const now = new Date();

    // Atomic single-op: the "no active extra" check lives INSIDE the filter,
    // so two concurrent purchases cannot both succeed. Active extra = remaining > 0
    // AND not expired. An exhausted (0) or expired pack passes the filter.
    // Aggregation-pipeline update copies subscriptionUntil -> extraBidsExpiry
    // inside the SAME operation — no race window between claim and expiry set.
    const res = await this.userModel.updateOne(
      {
        _id: id,
        role: UserRole.Worker,
        subscriptionActive: true,
        subscriptionUntil: { $gt: now },
        $or: [
          { extraBidsRemaining: { $lte: 0 } },
          { extraBidsRemaining: { $exists: false } },
          { extraBidsExpiry: { $lte: now } },
          { extraBidsExpiry: null },
          { extraBidsExpiry: { $exists: false } },
        ],
      },
      [
        {
          $set: {
            extraBidsRemaining: EXTRA_BIDS_PACK_SIZE,
            extraBidsExpiry: '$subscriptionUntil',
            extraBidsPurchasedAt: now,
            lastUpdated: now,
          },
        },
      ],
    ).exec();

    if (res.modifiedCount === 1) {
      const doc = await this.userModel.findById(id).exec();
      if (!doc) throw new NotFoundException(`Worker ${id} not found`);
      return doc;
    }

    // No match — read once to say WHY (mirrors consumeBid's error pattern)
    const w = await this.userModel.findOne({ _id: id, role: UserRole.Worker }).lean().exec() as { subscriptionActive?: boolean; subscriptionUntil?: Date | null; extraBidsRemaining?: number; extraBidsExpiry?: Date | null } | null;
    if (!w) throw new NotFoundException(`Worker ${id} not found`);
    if (w.subscriptionActive !== true || !w.subscriptionUntil || new Date(w.subscriptionUntil).getTime() <= now.getTime()) {
      throw new ForbiddenException('SUBSCRIPTION_REQUIRED');
    }
    throw new BadRequestException('EXTRA_ALREADY_ACTIVE');
  }

  async consumeBid(id: string): Promise<void> {
    const now   = new Date();
    const month = algiersMonthKey(now);
    const subscribed = {
      _id: id,
      role: UserRole.Worker,
      subscriptionActive: true,
      subscriptionUntil: { $gt: now },
    };

    // Extra bids consumed first — one active pack at a time, no accumulation.
    // Guard: extra valid only if not expired (extraBidsExpiry > now).
    const extraConsumed = await this.userModel.updateOne(
      { ...subscribed, extraBidsRemaining: { $gt: 0 }, extraBidsExpiry: { $gt: now } },
      { $inc: { extraBidsRemaining: -1 } },
    ).exec();
    if (extraConsumed.modifiedCount === 1) return;

    const quota = { $ifNull: ['$monthlyBidQuota', Number.MAX_SAFE_INTEGER] };

    // Fast path: same-month bucket with room left.
    const sameMonth = await this.userModel.updateOne(
      { ...subscribed, bidMonth: month,
        $expr: { $lt: [{ $ifNull: ['$bidsUsed', 0] }, quota] } },
      { $inc: { bidsUsed: 1 } },
    ).exec();
    if (sameMonth.modifiedCount === 1) return;

    // Month rollover: fresh bucket (quota must allow at least one bid).
    const rolled = await this.userModel.updateOne(
      { ...subscribed, bidMonth: { $ne: month }, $expr: { $lt: [0, quota] } },
      { $set: { bidsUsed: 1, bidMonth: month } },
    ).exec();
    if (rolled.modifiedCount === 1) return;

    // Neither matched — read once to say WHY (error drives the app's CTA).
    const w = await this.userModel
      .findOne({ _id: id, role: UserRole.Worker })
      .select('subscriptionActive subscriptionUntil monthlyBidQuota')
      .lean()
      .exec() as {
        subscriptionActive?: boolean;
        subscriptionUntil?: Date | null;
        monthlyBidQuota?: number | null;
      } | null;
    if (
      !w ||
      w.subscriptionActive !== true ||
      w.subscriptionUntil == null ||
      new Date(w.subscriptionUntil).getTime() <= now.getTime()
    ) {
      throw new ForbiddenException('SUBSCRIPTION_REQUIRED');
    }
    if (w.monthlyBidQuota === 0) throw new ForbiddenException('BID_NOT_INCLUDED');
    throw new ForbiddenException('BID_QUOTA_EXHAUSTED');
  }

  /**
   * Replace the worker's portfolio with [urls] — the caller has already checked
   * that every URL is an image this worker uploaded to our Cloudinary account.
   * Returns the URLs that dropped out, so the caller can release their objects.
   *
   * Quota: `portfolioQuota` is written at activation. Documents that subscribed
   * before the field existed are grandfathered from `subscriptionPrice` (same
   * formula), so nobody has to renew to unlock what they paid for. No active
   * subscription → no portfolio, reusing the bid gate's SUBSCRIPTION_REQUIRED
   * code that the app already maps to its upgrade CTA.
   *
   * Replacement (not append) is what makes this safe without a lock: every
   * write carries the whole list and is length-checked, so concurrent writes
   * can lose an edit but can never overshoot the quota.
   */
  async setPortfolio(id: string, urls: string[]): Promise<string[]> {
    const now = new Date();
    const w = await this.userModel
      .findOne({ _id: id, role: UserRole.Worker })
      .select('portfolio portfolioQuota subscriptionPrice subscriptionActive subscriptionUntil')
      .lean()
      .exec() as {
        portfolio?: string[];
        portfolioQuota?: number | null;
        subscriptionPrice?: number | null;
        subscriptionActive?: boolean;
        subscriptionUntil?: Date | null;
      } | null;
    if (!w) throw new NotFoundException(`Worker ${id} not found`);

    const active =
      w.subscriptionActive === true &&
      w.subscriptionUntil != null &&
      new Date(w.subscriptionUntil).getTime() > now.getTime();
    const quota = active
      ? (w.portfolioQuota ?? portfolioQuotaForPrice(w.subscriptionPrice ?? 0))
      : 0;
    if (quota <= 0) throw new ForbiddenException('SUBSCRIPTION_REQUIRED');

    const next = [...new Set(urls)];
    if (next.length > quota) throw new ForbiddenException('PORTFOLIO_QUOTA_EXCEEDED');

    await this.userModel
      .updateOne(
        { _id: id, role: UserRole.Worker },
        { $set: { portfolio: next, lastUpdated: now } },
      )
      .exec();

    const kept = new Set(next);
    return (w.portfolio ?? []).filter((u) => !kept.has(u));
  }

  /** Return one bid to the bucket (compensation when a submit fails late).
   *  Refunds to extra if an extra pack was partially consumed (extraRemaining < PACK_SIZE),
   *  regardless of expiry — a bid consumed from extra must return there even if the
   *  pack expired between consume and refund. Otherwise refunds to the monthly bucket. */
  async refundBid(id: string): Promise<void> {
    const w = await this.userModel.findOne({ _id: id }).select('extraBidsRemaining extraBidsExpiry extraBidsPurchasedAt').lean().exec() as { extraBidsRemaining?: number; extraBidsExpiry?: Date | null; extraBidsPurchasedAt?: Date | null } | null;
    // Extra was partially consumed → refund there (even if expired — the bid came from extra).
    // Guarded $inc: concurrent refunds can never push remaining past PACK_SIZE.
    if (w && w.extraBidsPurchasedAt != null) {
      const r = await this.userModel.updateOne(
        { _id: id, extraBidsRemaining: { $lt: EXTRA_BIDS_PACK_SIZE } },
        { $inc: { extraBidsRemaining: 1 } },
      ).exec();
      if (r.modifiedCount === 1) {
        // If extra had expired, revive expiry so the refunded bid is usable
        const now = new Date();
        if (!w.extraBidsExpiry || new Date(w.extraBidsExpiry).getTime() <= now.getTime()) {
          const sub = await this.userModel.findById(id).select('subscriptionUntil').lean().exec() as { subscriptionUntil?: Date | null } | null;
          await this.userModel.updateOne({ _id: id }, { $set: { extraBidsExpiry: sub?.subscriptionUntil ?? null } }).exec();
        }
        return;
      }
    }
    await this.userModel
      .updateOne({ _id: id, bidsUsed: { $gt: 0 } }, { $inc: { bidsUsed: -1 } })
      .exec();
  }

  async updateWorker(id: string, dto: UpdateWorkerDto): Promise<UserDocument> {
    try {
      const patch: Partial<Record<string, unknown>> = { lastUpdated: new Date() };
      if (dto.name             != null) patch['name']            = dto.name;
      if (dto.phoneNumber      != null) patch['phoneNumber']     = dto.phoneNumber;
      if (dto.profileImageUrl  != null) patch['profileImageUrl'] = dto.profileImageUrl;
      if (dto.cellId           != null) patch['cellId']          = dto.cellId;
      if (dto.wilayaCode       != null) patch['wilayaCode']      = dto.wilayaCode;
      if (dto.geoHash          != null) patch['geoHash']         = dto.geoHash;
      if (dto.bio              != null) patch['bio']             = dto.bio;
      // averageRating / ratingCount / ratingSum / jobsCompleted / responseRate
      // are server-derived trust data (applyRating, completeJob) — the Flutter
      // app sends them in toMap() but they must NEVER be client-writable.
      if (dto.lastActiveAt     != null) patch['lastActiveAt']    = dto.lastActiveAt;

      const doc = await this.userModel
        .findOneAndUpdate(
          { _id: id, role: UserRole.Worker },
          patch,
          { new: true, runValidators: true },
        )
        .exec();
      if (!doc) throw new NotFoundException(`Worker ${id} not found`);
      this.vectorIndex.indexWorker(doc); // bio edits re-embed the profile
      return doc;
    } catch (err) {
      this.logger.error(`UsersService.updateWorker(${id}) failed`, err);
      throw err;
    }
  }

  async updateWorkerStatus(id: string, isOnline: boolean): Promise<void> {
    try {
      const now = new Date();
      const set: Record<string, unknown> = { isOnline, lastUpdated: now };
      const update: Record<string, unknown> = { $set: set };

      if (isOnline) {
        // Going online requires an active pack — mirrors the discovery filter
        // and the bid gate. Blocks direct-API abuse the UI paywall can't.
        const sub = (await this.userModel
          .findOne({ _id: id, role: UserRole.Worker })
          .select('subscriptionActive subscriptionUntil')
          .lean()
          .exec()) as
          | { subscriptionActive?: boolean; subscriptionUntil?: Date | null }
          | null;
        if (!sub) throw new NotFoundException(`Worker ${id} not found`);
        if (
          sub.subscriptionActive !== true ||
          sub.subscriptionUntil == null ||
          new Date(sub.subscriptionUntil).getTime() <= now.getTime()
        ) {
          throw new ForbiddenException('SUBSCRIPTION_REQUIRED');
        }
        set.onlineSince = now;
      } else {
        set.lastActiveAt = now;
        set.onlineSince = null;
        // Accumulate the session's elapsed online time into TODAY's bucket.
        // The daily counter resets at 00:00 (Africa/Algiers): only the part of
        // the session inside the current local day is credited — the pre-
        // midnight remainder belongs to a bucket nobody displays anymore.
        const w = await this.userModel
          .findOne({ _id: id, role: UserRole.Worker })
          .select('onlineSince usageDay')
          .lean()
          .exec();
        const prev = w as { onlineSince?: Date; usageDay?: string | null } | null;
        const since = prev?.onlineSince;
        const today = algiersDayKey(now);
        set.usageDay = today;
        if (since) {
          const elapsed = Math.floor((now.getTime() - new Date(since).getTime()) / 1000);
          const credit = Math.min(
            Math.max(elapsed, 0),
            secondsSinceAlgiersMidnight(now),
          );
          if (prev?.usageDay === today) {
            if (credit > 0) update.$inc = { usageSeconds: credit };
          } else {
            set.usageSeconds = credit; // day rolled over → fresh bucket
          }
        } else if (prev?.usageDay !== today) {
          set.usageSeconds = 0; // stale bucket from a previous day
        }
      }

      const result = await this.userModel
        .updateOne({ _id: id, role: UserRole.Worker }, update)
        .exec();
      if (result.matchedCount === 0) throw new NotFoundException(`Worker ${id} not found`);
    } catch (err) {
      this.logger.error(`UsersService.updateWorkerStatus(${id}) failed`, err);
      throw err;
    }
  }

  async updateWorkerLocation(
    id: string,
    latitude: number,
    longitude: number,
    cellId?: string,
    wilayaCode?: number,
    geoHash?: string,
  ): Promise<void> {
    try {
      const patch: Partial<Record<string, unknown>> = {
        latitude,
        longitude,
        lastUpdated: new Date(),
      };
      if (cellId     != null) { patch['cellId'] = cellId; patch['lastCellUpdate'] = new Date(); }
      if (wilayaCode != null) patch['wilayaCode'] = wilayaCode;
      if (geoHash    != null) patch['geoHash']    = geoHash;

      const result = await this.userModel
        .updateOne({ _id: id, role: UserRole.Worker }, patch)
        .exec();
      if (result.matchedCount === 0) throw new NotFoundException(`Worker ${id} not found`);
    } catch (err) {
      this.logger.error(`UsersService.updateWorkerLocation(${id}) failed`, err);
      throw err;
    }
  }

  async updateWorkerFcmToken(id: string, fcmToken: string): Promise<void> {
    try {
      const result = await this.userModel
        .updateOne({ _id: id, role: UserRole.Worker }, { fcmToken, lastUpdated: new Date() })
        .exec();
      if (result.matchedCount === 0) throw new NotFoundException(`Worker ${id} not found`);
    } catch (err) {
      this.logger.error(`UsersService.updateWorkerFcmToken(${id}) failed`, err);
      throw err;
    }
  }

  /** Server-side jobsCompleted counter — the only legitimate write path. */
  async incrementJobsCompleted(id: string): Promise<void> {
    try {
      await this.userModel
        .updateOne(
          { _id: id, role: UserRole.Worker },
          { $inc: { jobsCompleted: 1 }, $set: { lastUpdated: new Date() } },
        )
        .exec();
    } catch (err) {
      this.logger.error(`UsersService.incrementJobsCompleted(${id}) failed`, err);
      throw err;
    }
  }

  async applyRating(id: string, stars: number): Promise<void> {    try {
      // Update-pipeline form: count/sum accumulate with $add (atomic under
      // concurrency, unlike read-modify-write) and the Bayesian average is
      // recomputed from the same values in the same operation.
      const C = 3.5;
      const m = 10;
      const result = await this.userModel.updateOne(
        { _id: id, role: UserRole.Worker },
        [
          {
            $set: {
              ratingCount: { $add: [{ $ifNull: ['$ratingCount', 0] }, 1] },
              ratingSum:   { $add: [{ $ifNull: ['$ratingSum', 0] }, stars] },
            },
          },
          {
            $set: {
              averageRating: {
                $divide: [
                  { $add: [m * C, '$ratingSum'] },
                  { $add: [m, '$ratingCount'] },
                ],
              },
              lastUpdated: '$$NOW',
            },
          },
        ],
      ).exec();
      if (result.matchedCount === 0) throw new NotFoundException(`Worker ${id} not found`);
    } catch (err) {
      this.logger.error(`UsersService.applyRating(${id}) failed`, err);
      throw err;
    }
  }

  /** Lean worker-only fetch — null for any non-worker uid. Used by the WS
   *  gateway and by /search/requests (worker gate + profile-as-query text). */
  async getWorkerForGateway(
    uid: string,
  ): Promise<Pick<UserDocument, 'wilayaCode' | 'profession' | 'isOnline' | 'bio'> | null> {
    return this.userModel
      .findOne({ _id: uid, role: UserRole.Worker })
      .select('wilayaCode profession isOnline bio')
      .lean()
      .exec() as any;
  }
}
