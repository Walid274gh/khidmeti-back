import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as crypto from 'crypto';

/**
 * SATIM (Société d'Automatisation des Transactions Interbancaires) gateway adapter.
 *
 * Supports Algerian cards only: CIB and EDAHABIA (no Visa/Mastercard).
 * Two modes:
 *   - live: SATIM_API_URL + SATIM_MERCHANT_ID + SATIM_SECRET_KEY are set → real HTTP calls
 *   - stub: env missing → local simulation for dev/testing (auto-confirms after intent)
 *
 * Interface is intentionally narrow so swapping to another Algerian PSP is one file.
 */

export interface CreateOrderResult {
  gatewayRef: string;
  redirectUrl: string | null; // SATIM hosted page URL
  raw: Record<string, unknown>;
}

export interface VerifyResult {
  status: 'confirmed' | 'failed' | 'pending';
  gatewayRef: string;
  raw: Record<string, unknown>;
}

@Injectable()
export class SatimGateway {
  private readonly logger = new Logger(SatimGateway.name);
  private readonly enabled: boolean;
  private readonly baseUrl: string;
  private readonly merchantId: string;
  private readonly secret: string;

  constructor(private readonly config: ConfigService) {
    this.baseUrl = this.config.get<string>('SATIM_API_URL', '');
    this.merchantId = this.config.get<string>('SATIM_MERCHANT_ID', '');
    this.secret = this.config.get<string>('SATIM_SECRET_KEY', '');
    this.enabled = !!(this.baseUrl && this.merchantId && this.secret);
    if (!this.enabled) this.logger.warn('SATIM env missing — running in STUB mode (dev only)');
  }

  isLive(): boolean { return this.enabled; }

  /** Create a payment order with SATIM. Returns hosted-page URL. */
  async createOrder(opts: {
    amount: number; // DZD, integer
    orderId: string; // our payment _id
    method: 'CIB' | 'EDAHABIA';
    description: string;
  }): Promise<CreateOrderResult> {
    if (!this.enabled) {
      return {
        gatewayRef: `stub_${opts.orderId}`,
        redirectUrl: null,
        raw: { stub: true, ...opts },
      };
    }
    // SATIM JSON API — adapt field names to actual SATIM spec when credentials arrive
    const body = {
      merchant_id: this.merchantId,
      order_id: opts.orderId,
      amount: opts.amount * 100, // SATIM expects centimes
      currency: 'DZD',
      description: opts.description,
      card_type: opts.method, // CIB | EDAHABIA
    };
    const sig = this.sign(JSON.stringify(body));
    const res = await fetch(`${this.baseUrl}/orders`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Signature': sig },
      body: JSON.stringify(body),
    });
    const raw = await res.json() as Record<string, unknown>;
    if (!res.ok) throw new Error(`SATIM createOrder failed: ${JSON.stringify(raw)}`);
    return {
      gatewayRef: (raw['order_id'] ?? raw['id'] ?? opts.orderId) as string,
      redirectUrl: (raw['redirect_url'] ?? raw['payment_url'] ?? null) as string | null,
      raw,
    };
  }

  /** Verify order status with SATIM */
  async verifyOrder(gatewayRef: string): Promise<VerifyResult> {
    if (!this.enabled) {
      // Stub: if ref starts with stub_ → confirmed (dev flow)
      return { status: gatewayRef.startsWith('stub_') ? 'confirmed' : 'pending', gatewayRef, raw: { stub: true } };
    }
    const sig = this.sign(gatewayRef);
    const res = await fetch(`${this.baseUrl}/orders/${encodeURIComponent(gatewayRef)}`, {
      headers: { 'X-Signature': sig },
    });
    const raw = await res.json() as Record<string, unknown>;
    const s = String(raw['status'] ?? '').toLowerCase();
    const status: VerifyResult['status'] =
      s === 'paid' || s === 'confirmed' || s === 'success' ? 'confirmed'
      : s === 'failed' || s === 'cancelled' ? 'failed' : 'pending';
    return { status, gatewayRef, raw };
  }

  /** Verify webhook HMAC — constant-time compare */
  verifyWebhookSignature(payload: string, signature: string): boolean {
    if (!this.secret) return false;
    const expected = this.sign(payload);
    if (expected.length !== signature.length) return false;
    return crypto.timingSafeEqual(Buffer.from(expected), Buffer.from(signature));
  }

  private sign(data: string): string {
    return crypto.createHmac('sha256', this.secret).update(data).digest('hex');
  }
}
