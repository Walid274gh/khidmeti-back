// apps/api/src/modules/users/users.controller.ts
//
// FIX Bug 2 — لا تنشئ stub تلقائياً عند GET /users/:id.
//
// المشكلة السابقة:
//   findById() كانت تستدعي ensureExists() والتي تنشئ وثيقة client في MongoDB
//   فور أول طلب GET. هذا يتعارض مع تسجيل العامل:
//     1. phone auth → Flutter تستدعي GET /users/:id (لفحص الملف)
//     2. ensureExists() تنشئ stub باسم "User" ودور "client"
//     3. تسجيل العامل يفشل (Bug 1) → يُغلق التطبيق
//     4. عند إعادة الفتح: الـ stub موجودة → role=client → يذهب للـ home
//
// الحل:
//   استخدم findById() الصارمة — ترجع 404 إذا لم توجد الوثيقة.
//   Flutter تعامل 404 كـ null في getUser() بشكل صحيح.
//   المستخدمون الجدد: 404 → null → checkAuthUser يُعيد isNewUser=true → setup flow
//   المستخدمون العائدون: وثيقتهم موجودة → fetch طبيعي
//
// ملاحظة: إذا احتجت يوماً لـ auto-provisioning، افعلها في POST /users فقط.

import {
  BadRequestException,
  Body,
  Controller,
  ForbiddenException,
  Get,
  HttpCode,
  HttpStatus,
  Logger,
  Param,
  Patch,
  Post,
  UseGuards,
} from '@nestjs/common';
import { FirebaseAuthGuard, SkipApprovalGate } from '../../common/guards/firebase-auth.guard';
import { CurrentUser }       from '../../common/decorators/current-user.decorator';
import { AuthUser }          from '../../common/guards/firebase-auth.guard';
import { UsersService }      from './users.service';
import { CreateUserDto }     from '../../dto/create-user.dto';
import { UpdateUserDto }     from '../../dto/update-user.dto';
import { UpdateLocationDto } from '../../dto/update-location.dto';
import { UpdateFcmTokenDto } from '../../dto/update-fcm-token.dto';
import { UpdateNotificationPrefsDto } from '../../dto/update-notification-prefs.dto';
import { UpdateAiConsentDto }         from '../../dto/update-ai-consent.dto';
import { UserDocument, SUBSCRIPTION_TIERS, SubscriptionTier } from '../../schemas/user.schema';

@Controller('users')
@UseGuards(FirebaseAuthGuard)
export class UsersController {
  private readonly logger = new Logger(UsersController.name);

  constructor(private readonly usersService: UsersService) {}

  /** POST /users — create or update caller's profile (client or worker). */
  // @SkipApprovalGate: this is the submission endpoint itself — a rejected
  // account resubmits its corrected documents through here.
  @Post()
  @SkipApprovalGate()
  @HttpCode(HttpStatus.OK)
  async upsert(
    @Body() dto: CreateUserDto,
    @CurrentUser() user: AuthUser,
  ): Promise<UserDocument> {
    if (dto.id !== user.uid) throw new ForbiddenException('You can only create your own profile');
    return this.usersService.upsert(dto);
  }

  /**
   * GET /users/:id
   *
   * FIX: يستخدم findById() الصارمة — لا auto-provisioning.
   * ترجع 404 إذا لم توجد الوثيقة. Flutter تعامل هذا كـ null.
   *
   * السبب: ensureExists() كانت تنشئ stub من نوع client قبل أن يختار
   * المستخدم دوره، مما يُفسد flow تسجيل العامل.
   */
  @Get(':id')
  async findById(
    @Param('id') id: string,
    @CurrentUser() _user: AuthUser,
  ): Promise<UserDocument> {
    return this.usersService.findById(id);
  }

  @Patch(':id')
  async update(
    @Param('id') id: string,
    @Body() dto: UpdateUserDto,
    @CurrentUser() user: AuthUser,
  ): Promise<UserDocument> {
    if (id !== user.uid) throw new ForbiddenException('You can only update your own profile');
    return this.usersService.update(id, dto);
  }

  @Patch(':id/location')
  @HttpCode(HttpStatus.NO_CONTENT)
  async updateLocation(
    @Param('id') id: string,
    @Body() dto: UpdateLocationDto,
    @CurrentUser() user: AuthUser,
  ): Promise<void> {
    if (id !== user.uid) throw new ForbiddenException('You can only update your own location');
    return this.usersService.updateLocation(
      id, dto.latitude, dto.longitude, dto.cellId, dto.wilayaCode, dto.geoHash,
    );
  }

  @Patch(':id/fcm-token')
  // @SkipApprovalGate: a pending/rejected account must still register its push
  // token — the "you have been approved" FCM notification depends on it.
  @SkipApprovalGate()
  @HttpCode(HttpStatus.NO_CONTENT)
  async updateFcmToken(
    @Param('id') id: string,
    @Body() dto: UpdateFcmTokenDto,
    @CurrentUser() user: AuthUser,
  ): Promise<void> {
    if (id !== user.uid) throw new ForbiddenException('You can only update your own FCM token');
    // Empty string = clear (sign-out / push disabled / account deletion).
    // clearFcmToken is idempotent and never 404s — a sign-out must not fail
    // because the profile was never created (abandoned onboarding, guests).
    if (dto.fcmToken.trim() === '') return this.usersService.clearFcmToken(id);
    return this.usersService.updateFcmToken(id, dto.fcmToken);
  }

  @Patch(':id/notification-prefs')
  @HttpCode(HttpStatus.NO_CONTENT)
  async updateNotificationPrefs(
    @Param('id') id: string,
    @Body() dto: UpdateNotificationPrefsDto,
    @CurrentUser() user: AuthUser,
  ): Promise<void> {
    if (id !== user.uid) {
      throw new ForbiddenException('You can only update your own notification preferences');
    }
    return this.usersService.updateNotificationPrefs(id, dto.mutedNotifTypes);
  }

  /** C1 flywheel — opt-in consent to log AI queries as training data. */
  @Patch(':id/ai-consent')
  @HttpCode(HttpStatus.NO_CONTENT)
  async updateAiConsent(
    @Param('id') id: string,
    @Body() dto: UpdateAiConsentDto,
    @CurrentUser() user: AuthUser,
  ): Promise<void> {
    if (id !== user.uid) {
      throw new ForbiddenException('You can only update your own AI-data consent');
    }
    return this.usersService.updateAiConsent(id, dto.aiDataConsent);
  }

  /**
   * POST /users/:id/subscription/activate
   *
   * Activates the worker's visibility subscription on a preset pack
   * (basic | pro | business | expert) or a slider-built custom pack
   * ({tier:'custom', hoursPerDay, bidsPerMonth, priority?, b2b?} — server
   * reprices, never trusts a client-sent price). Expert / custom-with-B2B
   * require admin-verified documents (403 DOCS_REQUIRED_FOR_B2B otherwise).
   * Back-compat: `{b2b:true}` maps to expert; unknown/missing → business.
   * ponytail: stub payment — the service marks the subscription active without
   * calling any gateway. Swap for SATIM callback verification when available.
   */
  @Post(':id/subscription/activate')
  @HttpCode(HttpStatus.OK)
  async activateSubscription(
    @Param('id') id: string,
    @Body() body: {
      tier?: string; b2b?: boolean;
      hoursPerDay?: number; bidsPerMonth?: number; priority?: boolean;
    },
    @CurrentUser() user: AuthUser,
  ): Promise<UserDocument> {
    if (id !== user.uid) throw new ForbiddenException('You can only subscribe your own account');
    const tier = (SUBSCRIPTION_TIERS as readonly string[]).includes(body?.tier ?? '')
      ? (body.tier as SubscriptionTier)
      : body?.b2b === true
        ? 'expert'
        : 'business';
    const custom = tier === 'custom'
      ? {
          hoursPerDay:  Number(body?.hoursPerDay),
          bidsPerMonth: Number(body?.bidsPerMonth),
          priority:     body?.priority === true,
          b2b:          body?.b2b === true,
        }
      : undefined;
    if (custom && (!Number.isFinite(custom.hoursPerDay) || !Number.isFinite(custom.bidsPerMonth))) {
      throw new BadRequestException('custom pack requires numeric hoursPerDay and bidsPerMonth');
    }
    return this.usersService.activateSubscription(id, tier, 30, custom);
  }
}
