// ══════════════════════════════════════════════════════════════════════════════
// AdminGuard
//
// Runs AFTER FirebaseAuthGuard (which sets request.user from the verified ID
// token). This guard loads the caller's MongoDB profile and enforces that its
// `role` is among the allowed roles — defaulting to 'admin' when @Roles() is
// absent — and that the account is not banned.
//
// DESIGN — why load the role from Mongo rather than a Firebase custom claim:
//   The platform's source of truth for role is the `users` collection (see
//   user.schema.ts). Mirroring AuthService, we inject UserModel directly (the
//   @Global DatabaseModule exports it) so no circular import with UsersModule.
//   promote-admin.ts also sets a Firebase custom claim as a convenience, but
//   the DB role is authoritative here.
//
// Usage:
//   @UseGuards(FirebaseAuthGuard, AdminGuard)   // admin-only (default)
//   @UseGuards(FirebaseAuthGuard, AdminGuard) @Roles(UserRole.Admin)
// ══════════════════════════════════════════════════════════════════════════════

import {
  CanActivate,
  ExecutionContext,
  ForbiddenException,
  Injectable,
  UnauthorizedException,
} from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { Request } from 'express';
import { User, UserDocument, UserRole } from '../../schemas/user.schema';
import { ROLES_KEY } from '../decorators/roles.decorator';

@Injectable()
export class AdminGuard implements CanActivate {
  constructor(
    private readonly reflector: Reflector,
    @InjectModel(User.name)
    private readonly userModel: Model<UserDocument>,
  ) {}

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const request = context.switchToHttp().getRequest<Request>();
    const uid = request.user?.uid;

    if (!uid) {
      throw new UnauthorizedException('No authenticated user on request');
    }

    const allowed =
      this.reflector.getAllAndOverride<UserRole[]>(ROLES_KEY, [
        context.getHandler(),
        context.getClass(),
      ]) ?? [UserRole.Admin];

    const doc = await this.userModel
      .findById(uid)
      .select('role isBanned')
      .lean()
      .exec();

    if (!doc) {
      throw new ForbiddenException('No profile found for this account');
    }
    if ((doc as unknown as { isBanned?: boolean }).isBanned) {
      throw new ForbiddenException('Account is banned');
    }

    const role = (doc as unknown as { role: UserRole }).role;
    if (!allowed.includes(role)) {
      throw new ForbiddenException('Insufficient privileges');
    }

    return true;
  }
}
