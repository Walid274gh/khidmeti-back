// ══════════════════════════════════════════════════════════════════════════════
// CreateWorkerDto
//
// Extends CreateUserDto and enforces role = 'worker'.
// The backend writes to the unified 'users' collection — no separate collection.
// ══════════════════════════════════════════════════════════════════════════════

import { IsString, IsNotEmpty, IsBoolean, IsOptional, MaxLength } from 'class-validator';
import { CreateUserDto } from './create-user.dto';
import { UserRole } from '../schemas/user.schema';

export class CreateWorkerDto extends CreateUserDto {
  /**
   * Always 'worker' — overrides the parent default of 'client'.
   * NestJS ValidationPipe strips this if submitted as anything else
   * because the controller's service enforces it server-side.
   */
  readonly role: UserRole = UserRole.Worker;

  @IsString()
  @IsNotEmpty()
  profession: string;

  @IsBoolean()
  @IsOptional()
  isOnline?: boolean;

  /**
   * Public "about me". '' is a meaningful value (the worker cleared it), so the
   * write path checks for undefined, not falsiness.
   */
  @IsString()
  @IsOptional()
  @MaxLength(300)
  bio?: string;
}
