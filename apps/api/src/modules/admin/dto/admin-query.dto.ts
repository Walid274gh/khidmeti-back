// ══════════════════════════════════════════════════════════════════════════════
// Admin query & mutation DTOs
//
// Validated with class-validator, matching the project convention. Query params
// arrive as strings; @Type/enableImplicitConversion (global ValidationPipe)
// coerces page/limit to numbers.
// ══════════════════════════════════════════════════════════════════════════════

import {
  IsBoolean,
  IsEnum,
  IsIn,
  IsInt,
  IsOptional,
  IsString,
  IsUrl,
  IsArray,
  ArrayMaxSize,
  Max,
  MaxLength,
  Min,
} from 'class-validator';
import { Transform, Type } from 'class-transformer';
import { UserRole } from '../../../schemas/user.schema';
import { ServiceStatus, BidStatus } from '../../../common/enums';

/**
 * Parse a query-string boolean. `Boolean('false')` is truthy, so we must map
 * explicitly — mirrors the `str === 'true'` idiom used elsewhere in the API.
 */
const toBool = ({ value }: { value: unknown }): boolean | undefined => {
  if (value === undefined || value === null || value === '') return undefined;
  if (typeof value === 'boolean') return value;
  return value === 'true' || value === '1';
};

/** Shared pagination + free-text search base. */
export class PaginationQueryDto {
  @IsOptional()
  @Type(() => Number)
  @IsInt()
  @Min(1)
  page?: number = 1;

  @IsOptional()
  @Type(() => Number)
  @IsInt()
  @Min(1)
  @Max(100)
  limit?: number = 20;

  @IsOptional()
  @IsString()
  @MaxLength(120)
  search?: string;

  @IsOptional()
  @IsString()
  sort?: string; // e.g. 'createdAt:desc'
}

export class ListUsersQueryDto extends PaginationQueryDto {
  @IsOptional()
  @IsEnum(UserRole)
  role?: UserRole;

  @IsOptional()
  @Transform(toBool)
  @IsBoolean()
  isBanned?: boolean;

  /** Filter the verifications queue: 'pending' | 'rejected'. */
  @IsOptional()
  @IsIn(['pending', 'rejected', 'approved'])
  verificationStatus?: string;
}

export class ListWorkersQueryDto extends PaginationQueryDto {
  @IsOptional()
  @IsString()
  profession?: string;

  @IsOptional()
  @Type(() => Number)
  @IsInt()
  wilayaCode?: number;

  @IsOptional()
  @Transform(toBool)
  @IsBoolean()
  isOnline?: boolean;

  @IsOptional()
  @Transform(toBool)
  @IsBoolean()
  isVerified?: boolean;
}

export class ListRequestsQueryDto extends PaginationQueryDto {
  @IsOptional()
  @IsEnum(ServiceStatus)
  status?: ServiceStatus;

  @IsOptional()
  @Type(() => Number)
  @IsInt()
  wilayaCode?: number;
}

export class ListBidsQueryDto extends PaginationQueryDto {
  @IsOptional()
  @IsEnum(BidStatus)
  status?: BidStatus;

  @IsOptional()
  @IsString()
  workerId?: string;

  @IsOptional()
  @IsString()
  serviceRequestId?: string;
}

// ── Mutations ────────────────────────────────────────────────────────────────

export class UpdateUserAdminDto {
  @IsOptional()
  @IsString()
  @MaxLength(120)
  name?: string;

  @IsOptional()
  @IsString()
  @MaxLength(160)
  email?: string;

  @IsOptional()
  @IsString()
  @MaxLength(30)
  phoneNumber?: string;

  @IsOptional()
  @IsString()
  profession?: string;

  /**
   * Moderation takedown for the public showcase gallery: an admin sends the
   * trimmed list (or []) to pull an inappropriate photo. Workers own their
   * gallery through PUT /workers/:id/portfolio; this is the only other way in,
   * and it is the reason a reported photo does not require banning the account.
   */
  @IsOptional()
  @IsArray()
  @ArrayMaxSize(50)
  @IsUrl({}, { each: true })
  portfolio?: string[];
}

export class SetRoleDto {
  @IsEnum(UserRole)
  role!: UserRole;
}

export class SetBanDto {
  @IsBoolean()
  isBanned!: boolean;
}

export class SetVerifiedDto {
  @IsBoolean()
  isVerified!: boolean;
}

/** Approve or reject a submitted verification-document set. */
export class SetVerificationDto {
  @IsIn(['approved', 'rejected'])
  status!: 'approved' | 'rejected';

  /** Optional note shown to the user on rejection. */
  @IsOptional()
  @IsString()
  @MaxLength(500)
  note?: string;
}

export class SetOnlineDto {
  @IsBoolean()
  isOnline!: boolean;
}

// ── Broadcast ────────────────────────────────────────────────────────────────

export class BroadcastDto {
  @IsString()
  @MaxLength(120)
  title!: string;

  @IsString()
  @MaxLength(500)
  body!: string;

  /** Target audience segment. */
  @IsIn(['all', 'clients', 'workers', 'wilaya'])
  audience!: 'all' | 'clients' | 'workers' | 'wilaya';

  @IsOptional()
  @Type(() => Number)
  @IsInt()
  wilayaCode?: number;
}
