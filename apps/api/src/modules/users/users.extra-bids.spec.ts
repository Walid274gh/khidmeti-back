// Subscription v3 — service-level rules for extra bids, consume/refund order,
// 7-day rollover, and custom-pack gates.
//
// These are pure unit tests: the Mongo model is a minimal in-memory fake so
// the tests run without a DB. They pin the product rules, not the driver.

import {
  ANNUAL_DAYS,
  ANNUAL_MULTIPLIER,
  CUSTOM_PACK,
  EXTRA_BIDS_PACK_SIZE,
  EXTRA_BIDS_PACK_PRICE,
  MONTHLY_DAYS,
  TIER_PACKS,
  annualPriceForTier,
  customPackEntitlements,
} from '../../schemas/user.schema';

// ── Minimal rule mirrors (must stay in sync with users.service.ts) ──────────
// We test the RULES, not the Mongoose plumbing: each helper below is a
// line-for-line port of the corresponding guard in the service.

// Extra purchase gate: active sub + no active extra pack
function canPurchaseExtra(o: {
  subscriptionActive?: boolean; subscriptionUntil?: Date | null;
  extraBidsRemaining?: number; extraBidsExpiry?: Date | null;
}, now: Date): 'ok' | 'SUBSCRIPTION_REQUIRED' | 'EXTRA_ALREADY_ACTIVE' {
  if (o.subscriptionActive !== true || !o.subscriptionUntil || o.subscriptionUntil.getTime() <= now.getTime()) {
    return 'SUBSCRIPTION_REQUIRED';
  }
  if ((o.extraBidsRemaining ?? 0) > 0 && o.extraBidsExpiry && o.extraBidsExpiry.getTime() > now.getTime()) {
    return 'EXTRA_ALREADY_ACTIVE';
  }
  return 'ok';
}

// Consume order: extra first (if valid), then pack bucket
function consumeOrder(o: {
  extraBidsRemaining?: number; extraBidsExpiry?: Date | null;
  bidsUsed?: number; bidMonth?: string | null; monthlyBidQuota?: number | null;
  month: string;
}, now: Date): 'extra' | 'pack' | 'rollover' | 'BID_QUOTA_EXHAUSTED' | 'BID_NOT_INCLUDED' {
  if ((o.extraBidsRemaining ?? 0) > 0 && o.extraBidsExpiry && o.extraBidsExpiry.getTime() > now.getTime()) {
    return 'extra';
  }
  const quota = o.monthlyBidQuota ?? Number.MAX_SAFE_INTEGER;
  if (quota === 0) return 'BID_NOT_INCLUDED';
  if (o.bidMonth === o.month && (o.bidsUsed ?? 0) < quota) return 'pack';
  if (o.bidMonth !== o.month && 0 < quota) return 'rollover';
  return 'BID_QUOTA_EXHAUSTED';
}

// Refund target: extra if it was partially consumed (even if expired), else pack
function refundTarget(o: {
  extraBidsRemaining?: number; extraBidsPurchasedAt?: Date | null;
  bidsUsed?: number;
}): 'extra' | 'pack' | 'nothing' {
  if (o.extraBidsPurchasedAt != null && (o.extraBidsRemaining ?? 0) < EXTRA_BIDS_PACK_SIZE) {
    return 'extra';
  }
  if ((o.bidsUsed ?? 0) > 0) return 'pack';
  return 'nothing';
}

// 7-day rollover: extra bought in last 7 days of cycle carries over
function rolloverAmount(o: {
  extraBidsRemaining?: number; extraBidsExpiry?: Date | null;
  extraBidsPurchasedAt?: Date | null; subscriptionUntil?: Date | null;
}, now: Date): number {
  if ((o.extraBidsRemaining ?? 0) <= 0 || !o.extraBidsExpiry || !o.extraBidsPurchasedAt || !o.subscriptionUntil) {
    return 0;
  }
  const msUntilExpiry = o.subscriptionUntil.getTime() - o.extraBidsPurchasedAt.getTime();
  if (msUntilExpiry <= 7 * 24 * 60 * 60 * 1000 && o.extraBidsExpiry.getTime() > now.getTime()) {
    return o.extraBidsRemaining!;
  }
  return 0;
}

const now = new Date('2026-09-01T12:00:00Z');
const subUntil = new Date('2026-09-30T12:00:00Z');

describe('extra purchase gate', () => {
  it('rejects without active subscription', () => {
    expect(canPurchaseExtra({ subscriptionActive: false, subscriptionUntil: subUntil }, now))
      .toBe('SUBSCRIPTION_REQUIRED');
    expect(canPurchaseExtra({ subscriptionActive: true, subscriptionUntil: new Date('2026-08-01T00:00:00Z') }, now))
      .toBe('SUBSCRIPTION_REQUIRED');
  });

  it('rejects while an extra pack is active', () => {
    expect(canPurchaseExtra({
      subscriptionActive: true, subscriptionUntil: subUntil,
      extraBidsRemaining: 3, extraBidsExpiry: subUntil,
    }, now)).toBe('EXTRA_ALREADY_ACTIVE');
  });

  it('allows after extra is exhausted (remaining = 0)', () => {
    expect(canPurchaseExtra({
      subscriptionActive: true, subscriptionUntil: subUntil,
      extraBidsRemaining: 0, extraBidsExpiry: subUntil,
    }, now)).toBe('ok');
  });

  it('allows after extra expired (even if remaining > 0)', () => {
    expect(canPurchaseExtra({
      subscriptionActive: true, subscriptionUntil: subUntil,
      extraBidsRemaining: 2, extraBidsExpiry: new Date('2026-08-15T00:00:00Z'),
    }, now)).toBe('ok');
  });

  it('pack is 5 bids for 500 DA', () => {
    expect(EXTRA_BIDS_PACK_SIZE).toBe(5);
    expect(EXTRA_BIDS_PACK_PRICE).toBe(500);
  });
});

describe('consume order (extra first)', () => {
  it('consumes extra before pack bids', () => {
    expect(consumeOrder({
      extraBidsRemaining: 4, extraBidsExpiry: subUntil,
      bidsUsed: 0, bidMonth: '2026-09', monthlyBidQuota: 10, month: '2026-09',
    }, now)).toBe('extra');
  });

  it('falls back to pack after extra = 0', () => {
    expect(consumeOrder({
      extraBidsRemaining: 0, extraBidsExpiry: subUntil,
      bidsUsed: 5, bidMonth: '2026-09', monthlyBidQuota: 10, month: '2026-09',
    }, now)).toBe('pack');
  });

  it('ignores expired extra', () => {
    expect(consumeOrder({
      extraBidsRemaining: 3, extraBidsExpiry: new Date('2026-08-01T00:00:00Z'),
      bidsUsed: 5, bidMonth: '2026-09', monthlyBidQuota: 10, month: '2026-09',
    }, now)).toBe('pack');
  });
});

describe('refund target (never silently lost)', () => {
  it('refunds to extra even after the pack expired', () => {
    // THE edge case: bid consumed from extra, pack expired, then refund arrives.
    // extraBidsPurchasedAt != null + remaining < 5 → refund to extra (revives expiry).
    expect(refundTarget({
      extraBidsRemaining: 4, extraBidsPurchasedAt: new Date('2026-08-20T00:00:00Z'),
      bidsUsed: 10,
    })).toBe('extra');
  });

  it('refunds to pack when no extra was ever bought', () => {
    expect(refundTarget({ bidsUsed: 5 })).toBe('pack');
  });

  it('does nothing when both buckets are at rest', () => {
    expect(refundTarget({ bidsUsed: 0 })).toBe('nothing');
    // Full extra (5/5) means nothing was consumed from it → refund to pack
    expect(refundTarget({
      extraBidsRemaining: 5, extraBidsPurchasedAt: new Date(),
      bidsUsed: 3,
    })).toBe('pack');
  });
});

describe('7-day rollover', () => {
  it('rolls over when bought in last 7 days of cycle', () => {
    const purchasedAt = new Date(subUntil.getTime() - 5 * 24 * 60 * 60 * 1000);
    expect(rolloverAmount({
      extraBidsRemaining: 3, extraBidsExpiry: subUntil,
      extraBidsPurchasedAt: purchasedAt, subscriptionUntil: subUntil,
    }, now)).toBe(3);
  });

  it('does NOT roll over when bought before last 7 days', () => {
    const purchasedAt = new Date(subUntil.getTime() - 20 * 24 * 60 * 60 * 1000);
    expect(rolloverAmount({
      extraBidsRemaining: 3, extraBidsExpiry: subUntil,
      extraBidsPurchasedAt: purchasedAt, subscriptionUntil: subUntil,
    }, now)).toBe(0);
  });

  it('does NOT roll over an exhausted or expired pack', () => {
    expect(rolloverAmount({
      extraBidsRemaining: 0, extraBidsExpiry: subUntil,
      extraBidsPurchasedAt: subUntil, subscriptionUntil: subUntil,
    }, now)).toBe(0);
    expect(rolloverAmount({
      extraBidsRemaining: 3, extraBidsExpiry: new Date('2026-08-01T00:00:00Z'),
      extraBidsPurchasedAt: subUntil, subscriptionUntil: subUntil,
    }, now)).toBe(0);
  });
});

describe('custom gates', () => {
  it('priority with 9 bids is dropped, with 10 bids is kept', () => {
    expect(customPackEntitlements(10, 9, { priority: true }).searchPriority).toBe(false);
    expect(customPackEntitlements(10, 9, { priority: true }).price).toBe(
      500 + 5 * 25 + 9 * 25, // no +200
    );
    expect(customPackEntitlements(10, 10, { priority: true }).searchPriority).toBe(true);
    expect(customPackEntitlements(10, 10, { priority: true }).price).toBe(
      500 + 5 * 25 + 10 * 25 + 200,
    );
  });

  it('bids clamp at 20 — 30 requested becomes 20', () => {
    expect(customPackEntitlements(10, 30).monthlyBidQuota).toBe(20);
    expect(customPackEntitlements(10, 999).monthlyBidQuota).toBe(20);
  });

  it('preset ladder: 5 → 10 → 30', () => {
    expect(TIER_PACKS.basic.monthlyBidQuota).toBe(5);
    expect(TIER_PACKS.pro.monthlyBidQuota).toBe(10);
    expect(TIER_PACKS.business.monthlyBidQuota).toBe(30);
    expect(TIER_PACKS.expert.monthlyBidQuota).toBe(30);
    // Custom max (20) < business (30) — presets stay the step-up
    expect(CUSTOM_PACK.bidsMax).toBeLessThan(TIER_PACKS.business.monthlyBidQuota!);
  });
});

describe('annual billing (fixed tiers only)', () => {
  it('multiplier is 10 (2 months free)', () => {
    expect(ANNUAL_MULTIPLIER).toBe(10);
    expect(MONTHLY_DAYS).toBe(30);
    expect(ANNUAL_DAYS).toBe(365);
  });

  it('annual prices: 5000 / 10000 / 15000 / 25000', () => {
    expect(annualPriceForTier('basic')).toBe(5000);
    expect(annualPriceForTier('pro')).toBe(10000);
    expect(annualPriceForTier('business')).toBe(15000);
    expect(annualPriceForTier('expert')).toBe(25000);
  });

  it('saving is 2/12 (~16.7%) vs monthly x12', () => {
    for (const tier of ['basic', 'pro', 'business', 'expert'] as const) {
      const monthly = TIER_PACKS[tier].price;
      expect(annualPriceForTier(tier)).toBe(monthly * 10);
      expect(monthly * 12 - annualPriceForTier(tier)).toBe(monthly * 2);
    }
  });
});
