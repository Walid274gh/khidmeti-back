import {
  Body,
  Controller,
  ForbiddenException,
  Get,
  Headers,
  HttpCode,
  HttpStatus,
  Logger,
  Post,
  Query,
  UseGuards,
} from '@nestjs/common';
import { Throttle } from '@nestjs/throttler';
import { FirebaseAuthGuard, SkipPinGate, SkipApprovalGate }  from '../../common/guards/firebase-auth.guard';
import { CurrentUser }                     from '../../common/decorators/current-user.decorator';
import { AuthUser }                        from '../../common/guards/firebase-auth.guard';
import { AuthService, UserCheckResult }    from './auth.service';
import { PinGateService }                  from './pin-gate.service';
import { RemovePinDto, SetPinDto, VerifyPinDto } from '../../dto/pin.dto';

@Controller('auth')
@UseGuards(FirebaseAuthGuard)
export class AuthController {
  private readonly logger = new Logger(AuthController.name);

  constructor(
    private readonly authService: AuthService,
    private readonly pinGate: PinGateService,
  ) {}

  /**
   * GET /auth/check?uid=:uid
   *
   * Appelé immédiatement après Firebase signInWithCredential pour déterminer
   * si l'utilisateur authentifié possède déjà un profil backend.
   *
   * Réponse :
   *   • { isNewUser: true,  role: null, ... }     → rediriger vers /role-selection
   *   • { isNewUser: false, role: 'client', ... } → rediriger vers /home
   *   • { ..., pinRequired: true }                → écran PIN d'abord
   *
   * Sécurité :
   *   Le paramètre uid DOIT correspondre à user.uid du JWT.
   *   Un utilisateur ne peut pas interroger le statut d'un autre compte.
   *
   * @SkipPinGate : ce endpoint est la sonde qui DÉTECTE pinRequired — il doit
   * répondre même depuis un device non vérifié.
   *
   * Rate limiting :
   *   10 req/min par UID — empêche l'énumération et protège le compte Firebase billing.
   */
  @Get('check')
  @SkipPinGate()
  @SkipApprovalGate()
  @HttpCode(HttpStatus.OK)
  @Throttle({ default: { limit: 10, ttl: 60_000 } })
  async check(
    @Query('uid') uid: string,
    @CurrentUser() user: AuthUser,
    @Headers('x-device-id') deviceId: string | undefined,
  ): Promise<UserCheckResult> {
    if (!uid?.trim()) {
      throw new ForbiddenException('uid query parameter is required');
    }
    if (uid !== user.uid) {
      this.logger.warn(
        `Auth check UID mismatch — JWT uid=${user.uid} vs query uid=${uid}`,
      );
      throw new ForbiddenException('UID mismatch — you may only check your own account');
    }

    return this.authService.checkUser(uid, deviceId);
  }

  // ── Account PIN (optional, anti SIM-recycling) ──────────────────────────────

  /**
   * POST /auth/verify-pin { pin }
   * From an unknown device: verify the account PIN and trust this device.
   * 5/min throttle + server-side 5-attempt lock (PinGateService) — a 6-digit
   * space is only safe because of these two limits.
   */
  @Post('verify-pin')
  @SkipPinGate()
  @HttpCode(HttpStatus.OK)
  @Throttle({ default: { limit: 5, ttl: 60_000 } })
  async verifyPin(
    @Body() dto: VerifyPinDto,
    @CurrentUser() user: AuthUser,
    @Headers('x-device-id') deviceId: string | undefined,
  ): Promise<{ verified: boolean; reason?: string; attemptsLeft?: number }> {
    if (!deviceId?.trim()) {
      throw new ForbiddenException('X-Device-Id header is required');
    }
    const result = await this.pinGate.verify(user.uid, dto.pin, deviceId);
    if (result.ok) return { verified: true };
    return { verified: false, reason: result.reason, attemptsLeft: result.attemptsLeft };
  }

  /**
   * POST /auth/pin { pin, currentPin? }
   * Set or change the account PIN. Changing requires the current PIN, so a
   * thief with an unlocked phone can't silently rotate it. NOT @SkipPinGate:
   * setting a PIN from an untrusted device on a protected account is exactly
   * what the gate exists to block.
   */
  @Post('pin')
  @HttpCode(HttpStatus.OK)
  @Throttle({ default: { limit: 5, ttl: 60_000 } })
  async setPin(
    @Body() dto: SetPinDto,
    @CurrentUser() user: AuthUser,
    @Headers('x-device-id') deviceId: string | undefined,
  ): Promise<{ ok: boolean; reason?: string }> {
    if (!deviceId?.trim()) {
      throw new ForbiddenException('X-Device-Id header is required');
    }
    const { hasPin } = await this.pinGate.status(user.uid, deviceId);
    if (hasPin) {
      if (!dto.currentPin) return { ok: false, reason: 'current_pin_required' };
      const check = await this.pinGate.verify(user.uid, dto.currentPin, deviceId);
      if (!check.ok) return { ok: false, reason: check.reason };
    }
    await this.pinGate.setPin(user.uid, dto.pin, deviceId);
    return { ok: true };
  }

  /** POST /auth/pin/remove { currentPin } — disable the PIN. */
  @Post('pin/remove')
  @HttpCode(HttpStatus.OK)
  @Throttle({ default: { limit: 5, ttl: 60_000 } })
  async removePin(
    @Body() dto: RemovePinDto,
    @CurrentUser() user: AuthUser,
    @Headers('x-device-id') deviceId: string | undefined,
  ): Promise<{ ok: boolean; reason?: string }> {
    if (!deviceId?.trim()) {
      throw new ForbiddenException('X-Device-Id header is required');
    }
    const check = await this.pinGate.verify(user.uid, dto.currentPin, deviceId);
    if (!check.ok) return { ok: false, reason: check.reason };
    await this.pinGate.removePin(user.uid);
    return { ok: true };
  }

  /**
   * POST /auth/request-pin-reset — forgotten PIN. Starts the 7-day cooling
   * period (WhatsApp model); after it elapses the next verify clears the PIN.
   * No SMS recovery — SMS is the channel this feature distrusts.
   */
  @Post('request-pin-reset')
  @SkipPinGate()
  @HttpCode(HttpStatus.OK)
  @Throttle({ default: { limit: 3, ttl: 60_000 } })
  async requestPinReset(
    @CurrentUser() user: AuthUser,
  ): Promise<{ resetAt: string }> {
    const resetAt = await this.pinGate.requestReset(user.uid);
    return { resetAt: resetAt.toISOString() };
  }
}
