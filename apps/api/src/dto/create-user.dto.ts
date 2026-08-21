// apps/api/src/dto/create-user.dto.ts
//
// FIX Bug 1 (defensive): @Transform يحوّل '' إلى undefined قبل التحقق.
// @IsOptional() يتجاهل null/undefined فقط — لا يتجاهل السلسلة الفارغة ''.
// بدون هذا الـ transform، يُفشل @IsEmail() على '' رغم @IsOptional().

import { Transform } from 'class-transformer';
import {
  IsString, IsEmail, IsOptional, IsNumber, IsNotEmpty,
  IsEnum, IsIn, Matches, MinLength, MaxLength, Min, Max,
  IsArray, IsUrl, ArrayMaxSize,
} from 'class-validator';
import { UserRole } from '../schemas/user.schema';

export class CreateUserDto {
  @IsString()
  @IsNotEmpty()
  id: string;

  @IsString()
  @IsNotEmpty()
  @MinLength(2)
  @MaxLength(50)
  name: string;

  /**
   * FIX: @Transform يحوّل '' إلى undefined قبل تشغيل @IsEmail().
   * ضروري لمستخدمي Phone Auth الذين لا يملكون email.
   * بدونه: email='' → @IsOptional() لا يتجاهله → @IsEmail() يفشل → 400.
   */
  @Transform(({ value }: { value: unknown }) =>
    value === '' || value === null ? undefined : value
  )
  @IsEmail()
  @IsOptional()
  email?: string;

  /** Defaults to 'client'. Pass 'worker' when registering a worker account. */
  @IsEnum(UserRole)
  @IsOptional()
  role?: UserRole;

  /**
   * Algerian phone number E.164 (+213XXXXXXXXX) or local (0[5-7]XXXXXXXX).
   */
  @IsString()
  @IsOptional()
  @Matches(/^(\+213[5-7]\d{8}|0[5-7]\d{8})$/, {
    message: 'phoneNumber must be a valid Algerian number (+213XXXXXXXXX or 0XXXXXXXXX)',
  })
  phoneNumber?: string;

  @IsNumber()
  @IsOptional()
  @Min(-90)
  @Max(90)
  latitude?: number;

  @IsNumber()
  @IsOptional()
  @Min(-180)
  @Max(180)
  longitude?: number;

  @IsString()
  @IsOptional()
  profileImageUrl?: string;

  @IsString()
  @IsOptional()
  fcmToken?: string;

  /** UI language for server-rendered notifications. */
  @IsString()
  @IsOptional()
  @IsIn(['fr', 'ar', 'en'])
  language?: string;

  /**
   * Cloudinary URLs of legal/identity documents (PDF or image), uploaded by the
   * client before this POST. Presence flips the account to 'pending' review
   * (see UsersService.upsert). Empty/absent → no review needed (doc-less worker
   * or client). Business accounts MUST send at least one (enforced in upsert).
   */
  @IsArray()
  @IsOptional()
  @ArrayMaxSize(10)
  @IsUrl({}, { each: true })
  verificationDocs?: string[];
}
