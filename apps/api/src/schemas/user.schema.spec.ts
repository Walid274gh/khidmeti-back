// Check for the visibility + pack logic in user.schema.ts:
//   • Africa/Algiers day helpers (fixed UTC+1, no DST) — drive the 00:00 reset.
//   • customPackEntitlements — slider + add-on pricing, 500 floor / 2300 max (20 bids cap).
//   • subscriptionVisibilityFilter — the single Mongo filter both discovery
//     paths (search + map) spread in: active sub + daily quota not exhausted
//     (reads stored entitlement fields; all packs are 7/7).

import {
  algiersDayKey,
  algiersMonthKey,
  secondsSinceAlgiersMidnight,
  customPackEntitlements,
  portfolioQuotaForPrice,
  TIER_PACKS,
  subscriptionVisibilityFilter,
} from './user.schema';

describe('daily usage bucket helpers', () => {
  it('algiersDayKey shifts UTC to local (+1h)', () => {
    // 23:30 UTC = 00:30 next day in Algiers
    expect(algiersDayKey(new Date('2026-01-01T23:30:00Z'))).toBe('2026-01-02');
    expect(algiersDayKey(new Date('2026-01-01T12:00:00Z'))).toBe('2026-01-01');
    expect(algiersMonthKey(new Date('2026-01-31T23:30:00Z'))).toBe('2026-02');
  });

  it('secondsSinceAlgiersMidnight caps a cross-midnight session', () => {
    // 01:00 local (00:00 UTC) → 3600 s available since midnight
    expect(secondsSinceAlgiersMidnight(new Date('2026-01-01T00:00:00Z'))).toBe(3600);
    // exactly local midnight → 0
    expect(secondsSinceAlgiersMidnight(new Date('2025-12-31T23:00:00Z'))).toBe(0);
  });
});

describe('customPackEntitlements', () => {
  it('floor: 5 h / 0 bids / no add-ons = 500 DZD', () => {
    const e = customPackEntitlements(5, 0);
    expect(e.price).toBe(500);
    expect(e.dailyQuotaSeconds).toBe(5 * 3600);
    expect(e.monthlyBidQuota).toBe(0);
    expect(e.searchPriority).toBe(false);
    expect(e.b2bAccess).toBe(false);
  });

  it('sliders maxed: 15 h / 20 bids = 1250 DZD', () => {
    expect(customPackEntitlements(15, 20).price).toBe(1250);
  });

  it('add-ons: priority +200 (requires >=10 bids), b2b +850; fully loaded 2300 < expert 2500', () => {
    // priority with 0 bids is ignored — requires >=10 bids
    expect(customPackEntitlements(5, 0, { priority: true }).price).toBe(500);
    expect(customPackEntitlements(5, 10, { priority: true }).price).toBe(950);
    expect(customPackEntitlements(5, 0, { b2b: true }).price).toBe(1350);
    const full = customPackEntitlements(15, 20, { priority: true, b2b: true });
    expect(full.price).toBe(2300);
    expect(full.price).toBeLessThan(TIER_PACKS.expert.price);
    expect(full.searchPriority).toBe(true);
    expect(full.b2bAccess).toBe(true);
  });

  it('clamps out-of-range slider values to the floor/ceiling', () => {
    expect(customPackEntitlements(1, -5).price).toBe(500);
    expect(customPackEntitlements(99, 999).price).toBe(1250);
  });

  it('mid-range: 8 h / 10 bids = 500 + 75 + 250 = 825', () => {
    expect(customPackEntitlements(8, 10).price).toBe(825);
  });
});

describe('TIER_PACKS', () => {
  it('matches the pack table (bid quota is the paid axis)', () => {
    expect(TIER_PACKS.basic).toMatchObject({ price: 500,  monthlyBidQuota: 5,  searchPriority: false });
    expect(TIER_PACKS.pro).toMatchObject({ price: 1000, monthlyBidQuota: 10, searchPriority: false });
    expect(TIER_PACKS.business).toMatchObject({ price: 1500, monthlyBidQuota: 30, searchPriority: true, b2bAccess: false });
    expect(TIER_PACKS.expert).toMatchObject({ price: 2500, monthlyBidQuota: 30, searchPriority: true, b2bAccess: true });
  });
});

describe('portfolioQuotaForPrice', () => {
  it('grants 1 photo per 100 DZD of pack price', () => {
    expect(portfolioQuotaForPrice(TIER_PACKS.basic.price)).toBe(5);
    expect(portfolioQuotaForPrice(TIER_PACKS.pro.price)).toBe(10);
    expect(portfolioQuotaForPrice(TIER_PACKS.business.price)).toBe(15);
    expect(portfolioQuotaForPrice(TIER_PACKS.expert.price)).toBe(25);
  });

  it('covers custom packs on the same scale (floor 500 → 5, max 2300 → 23)', () => {
    expect(portfolioQuotaForPrice(customPackEntitlements(5, 0).price)).toBe(5);
    expect(portfolioQuotaForPrice(customPackEntitlements(8, 10).price)).toBe(8);
    expect(
      portfolioQuotaForPrice(
        customPackEntitlements(15, 20, { priority: true, b2b: true }).price,
      ),
    ).toBe(23);
  });

  it('no subscription (no price) grants nothing', () => {
    expect(portfolioQuotaForPrice(0)).toBe(0);
    expect(portfolioQuotaForPrice(Number.NaN)).toBe(0);
  });
});

describe('subscriptionVisibilityFilter', () => {
  const monday   = new Date('2026-07-20T11:00:00Z');
  const saturday = new Date('2026-07-18T11:00:00Z');

  const exprOf = (d: Date) =>
    JSON.stringify(subscriptionVisibilityFilter(d).$expr);

  it('always requires an active, unexpired subscription', () => {
    const f = subscriptionVisibilityFilter(monday);
    expect(f.subscriptionActive).toBe(true);
    expect(f.subscriptionUntil).toEqual({ $gt: monday });
  });

  it('quota clause reads the stored dailyQuotaSeconds (null = unlimited)', () => {
    expect(exprOf(monday)).toContain('$dailyQuotaSeconds');
    expect(exprOf(monday)).toContain(String(Number.MAX_SAFE_INTEGER));
  });

  it('all packs are 7/7 — no weekend/price clause, filter is day-independent', () => {
    expect(exprOf(saturday)).not.toContain('$subscriptionPrice');
    // Same shape on any day; only the embedded timestamps differ.
    expect(Object.keys(subscriptionVisibilityFilter(saturday)))
      .toEqual(Object.keys(subscriptionVisibilityFilter(monday)));
  });
});
