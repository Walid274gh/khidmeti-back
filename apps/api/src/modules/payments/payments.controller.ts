import { Body, Controller, Get, Headers, Post, UseGuards, ForbiddenException, Req } from '@nestjs/common';
import { FirebaseAuthGuard } from '../../common/guards/firebase-auth.guard';
import { CurrentUser, AuthUser } from '../../common/decorators/current-user.decorator';
import { PaymentsService } from './payments.service';
import { Request } from 'express';

@Controller('payments')
@UseGuards(FirebaseAuthGuard)
export class PaymentsController {
  constructor(private readonly payments: PaymentsService) {}

  /** POST /payments/intent — idempotent via Idempotency-Key header */
  @Post('intent')
  async createIntent(
    @Body() body: Record<string, unknown>,
    @CurrentUser() user: AuthUser,
    @Headers('idempotency-key') idempotencyKey?: string,
    @Headers('x-idempotency-key') xKey?: string,
  ) {
    const key = idempotencyKey ?? xKey;
    if (!key) throw new ForbiddenException('Idempotency-Key header required');
    const tier = String(body['tier'] ?? 'business');
    const method = String(body['method'] ?? 'CIB');
    return this.payments.createIntent(user.uid, tier, body, key, method);
  }

  /** POST /payments/webhook — SATIM callback (no auth, HMAC verified) */
  @Post('webhook')
  async webhook(@Req() req: Request) {
    const sig = (req.headers['x-signature'] ?? req.headers['x-satim-signature'] ?? '') as string;
    // raw body captured via express.raw middleware — fallback to JSON stringify
    const raw = (req as unknown as { rawBody?: string }).rawBody ?? JSON.stringify(req.body);
    return this.payments.handleWebhook(raw, sig);
  }

  @Get('me')
  async list(@CurrentUser() user: AuthUser) {
    return this.payments.listForUser(user.uid);
  }

  @Get('subscription')
  async subscription(@CurrentUser() user: AuthUser) {
    return this.payments.getSubscription(user.uid);
  }

  /** POST /payments/reconcile — manual trigger (admin/cron) */
  @Post('reconcile')
  async reconcile() {
    const fixed = await this.payments.reconcilePending();
    return { fixed };
  }
}
