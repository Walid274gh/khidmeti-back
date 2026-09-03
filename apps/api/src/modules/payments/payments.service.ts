import { BadRequestException, ForbiddenException, Injectable, Logger, NotFoundException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { randomUUID } from 'crypto';
import { Payment, PaymentDocument } from '../../schemas/payment.schema';
import { UsersService } from '../users/users.service';
import { SatimGateway } from './gateway/satim.gateway';
import {
  TIER_PACKS, customPackEntitlements, portfolioQuotaForPrice,
  SUBSCRIPTION_TIERS, SubscriptionTier, PackEntitlements, CUSTOM_PACK,
  ANNUAL_MULTIPLIER, ANNUAL_DAYS,
} from '../../schemas/user.schema';
import { InjectModel as InjectUserModel } from '@nestjs/mongoose';
import { User, UserDocument } from '../../schemas/user.schema';

const INTENT_TTL_MS = 15 * 60 * 1000; // 15 min

@Injectable()
export class PaymentsService {
  private readonly logger = new Logger(PaymentsService.name);

  constructor(
    @InjectModel(Payment.name) private readonly paymentModel: Model<PaymentDocument>,
    @InjectModel(User.name) private readonly userModel: Model<UserDocument>,
    private readonly usersService: UsersService,
    private readonly satim: SatimGateway,
  ) {}

  private entitlementsFor(tier: SubscriptionTier, body: Record<string, unknown>): PackEntitlements {
    if (tier === 'custom') {
      return customPackEntitlements(
        Number(body['hoursPerDay']), Number(body['bidsPerMonth']),
        { priority: body['priority'] === true, b2b: body['b2b'] === true },
      );
    }
    return TIER_PACKS[tier as Exclude<SubscriptionTier, 'custom'>];
  }

  /** Idempotent intent creation — same idempotencyKey returns existing doc */
  async createIntent(
    userId: string,
    tier: string,
    body: Record<string, unknown>,
    idempotencyKey: string,
    method: string,
  ): Promise<PaymentDocument> {
    if (!SUBSCRIPTION_TIERS.includes(tier as SubscriptionTier)) throw new BadRequestException('Invalid tier');
    if (method !== 'CIB' && method !== 'EDAHABIA') throw new BadRequestException('method must be CIB or EDAHABIA');
    if (!idempotencyKey || idempotencyKey.length < 8) throw new BadRequestException('Idempotency-Key required');

    // dedup on idempotencyKey — no second charge on double-tap
    const existing = await this.paymentModel.findOne({ idempotencyKey }).exec();
    if (existing) {
      if (existing.userId !== userId) throw new BadRequestException('Idempotency key belongs to another user');
      return existing;
    }

    const tierTyped = tier as SubscriptionTier;
    const period = body['period'] === 'annual' ? 'annual' : 'monthly';
    if (period === 'annual' && tierTyped === 'custom') {
      throw new BadRequestException('Annual billing is available for fixed tiers only');
    }
    if (tierTyped === 'custom') {
      if (!Number.isFinite(Number(body['hoursPerDay'])) || !Number.isFinite(Number(body['bidsPerMonth']))) {
        throw new BadRequestException('custom requires hoursPerDay and bidsPerMonth');
      }
    }
    const ent = this.entitlementsFor(tierTyped, body);
    const billedPrice = period === 'annual' && tierTyped !== 'custom'
      ? TIER_PACKS[tierTyped as Exclude<SubscriptionTier, 'custom'>].price * ANNUAL_MULTIPLIER
      : ent.price;
    const billedDays = period === 'annual' ? ANNUAL_DAYS : 30;

    // B2B gate check early — fail fast before charging
    if (ent.b2bAccess) {
      const u = await this.userModel.findById(userId).select('isVerified').lean().exec() as { isVerified?: boolean } | null;
      if (!u) throw new NotFoundException('User not found');
      if (u.isVerified !== true) throw new ForbiddenException('DOCS_REQUIRED_FOR_B2B');
    }

    const now = new Date();
    const paymentId = randomUUID();
    const gateway = await this.satim.createOrder({
      amount: billedPrice, orderId: paymentId, method: method as 'CIB' | 'EDAHABIA',
      description: `Khidmeti ${tierTyped} ${period} — ${billedPrice} DZD`,
    });

    const doc = await this.paymentModel.create({
      _id: paymentId,
      userId, tier: tierTyped, period, billedDays,
      entitlements: ent as unknown as Record<string, unknown>,
      amount: billedPrice, method, idempotencyKey,
      status: 'pending', gatewayRef: gateway.gatewayRef,
      gatewayResponse: gateway.raw,
      createdAt: now, expiresAt: new Date(now.getTime() + INTENT_TTL_MS),
    });

    // Stub mode: auto-confirm synchronously (dev only) — in live mode webhook does it
    if (!this.satim.isLive()) {
      await this.confirmPayment(doc._id);
      return (await this.paymentModel.findById(doc._id).exec())!;
    }
    return doc;
  }

  /** Confirm payment and activate entitlements atomically */
  async confirmPayment(paymentId: string): Promise<void> {
    const p = await this.paymentModel.findById(paymentId).exec();
    if (!p) throw new NotFoundException('Payment not found');
    if (p.status === 'confirmed') return; // idempotent
    if (p.status !== 'pending') throw new BadRequestException(`Cannot confirm payment in ${p.status} state`);

    const ent = p.entitlements as unknown as PackEntitlements;
    const now = new Date();
    const stored = p as unknown as { billedDays?: number; period?: string; amount?: number };
    const days = Number.isFinite(stored.billedDays) && (stored.billedDays as number) > 0 ? (stored.billedDays as number) : 30;
    const until = new Date(now.getTime() + days * 24 * 60 * 60 * 1000);
    const isAnnual = stored.period === 'annual';

    // Atomic: mark payment confirmed + activate subscription in same logical tx
    // Use findByIdAndUpdate for each — if second fails, reconciliation job retries
    await this.paymentModel.updateOne({ _id: p._id, status: 'pending' }, {
      status: 'confirmed', confirmedAt: now,
    }).exec();

    const tier = p.tier as SubscriptionTier;
    // Reuse stored entitlements — never recompute from tier table to avoid drift
    await this.userModel.findByIdAndUpdate(p.userId, {
      subscriptionActive: true,
      subscriptionUntil: until,
      subscriptionTier: tier,
      subscriptionPrice: typeof stored.amount === 'number' ? stored.amount : ent.price,
      subscriptionPeriod: isAnnual ? 'annual' : 'monthly',
      subscriptionAnnual: isAnnual,
      dailyQuotaSeconds: ent.dailyQuotaSeconds,
      monthlyBidQuota: ent.monthlyBidQuota,
      searchPriority: ent.searchPriority,
      b2bAccess: ent.b2bAccess,
      portfolioQuota: portfolioQuotaForPrice(ent.price),
      bidsUsed: 0,
      bidMonth: now.toISOString().slice(0, 7),
      lastUpdated: now,
    }).exec();
  }

  /** Webhook handler — verifies HMAC then confirms */
  async handleWebhook(rawBody: string, signature: string): Promise<{ ok: boolean }> {
    if (!this.satim.verifyWebhookSignature(rawBody, signature)) {
      throw new ForbiddenException('Invalid webhook signature');
    }
    const body = JSON.parse(rawBody) as Record<string, unknown>;
    const gatewayRef = String(body['order_id'] ?? body['gatewayRef'] ?? '');
    const p = await this.paymentModel.findOne({ gatewayRef }).exec();
    if (!p) throw new NotFoundException('Payment not found for gatewayRef');
    const s = String(body['status'] ?? '').toLowerCase();
    if (s === 'paid' || s === 'confirmed' || s === 'success') {
      await this.confirmPayment(p._id);
    } else if (s === 'failed' || s === 'cancelled') {
      await this.paymentModel.updateOne({ _id: p._id, status: 'pending' }, { status: 'failed' }).exec();
    }
    return { ok: true };
  }

  /** Reconcile: verify pending intents with gateway; expire stale ones */
  async reconcilePending(): Promise<number> {
    const cutoff = new Date(Date.now() - INTENT_TTL_MS);
    const pendings = await this.paymentModel.find({ status: 'pending', createdAt: { $lte: cutoff } }).limit(50).exec();
    let fixed = 0;
    for (const p of pendings) {
      try {
        if (p.expiresAt && p.expiresAt.getTime() < Date.now()) {
          await this.paymentModel.updateOne({ _id: p._id, status: 'pending' }, { status: 'expired' }).exec();
          continue;
        }
        if (p.gatewayRef) {
          const v = await this.satim.verifyOrder(p.gatewayRef);
          if (v.status === 'confirmed') { await this.confirmPayment(p._id); fixed++; }
          else if (v.status === 'failed') await this.paymentModel.updateOne({ _id: p._id }, { status: 'failed' }).exec();
        }
      } catch (e) { this.logger.warn(`reconcile ${p._id} failed: ${(e as Error).message}`); }
    }
    // Also fix subscriptions where payment is confirmed but user doc still inactive (crash during activation)
    const confirmed = await this.paymentModel.find({ status: 'confirmed' }).sort({ createdAt: -1 }).limit(100).exec();
    for (const p of confirmed) {
      const u = await this.userModel.findById(p.userId).select('subscriptionActive subscriptionUntil').lean().exec() as { subscriptionActive?: boolean; subscriptionUntil?: Date | null } | null;
      if (u && (u.subscriptionActive !== true || !u.subscriptionUntil || new Date(u.subscriptionUntil).getTime() <= Date.now())) {
        try { await this.confirmPayment(p._id); fixed++; } catch {}
      }
    }
    return fixed;
  }

  async listForUser(userId: string): Promise<PaymentDocument[]> {
    return this.paymentModel.find({ userId }).sort({ createdAt: -1 }).limit(50).exec();
  }

  async getSubscription(userId: string): Promise<Record<string, unknown> | null> {
    const u = await this.userModel.findById(userId).select('subscriptionActive subscriptionUntil subscriptionTier subscriptionPrice subscriptionPeriod subscriptionAnnual dailyQuotaSeconds monthlyBidQuota searchPriority b2bAccess portfolioQuota').lean().exec();
    return u as unknown as Record<string, unknown> | null;
  }
}
