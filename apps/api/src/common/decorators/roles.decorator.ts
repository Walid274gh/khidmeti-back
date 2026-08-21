// ══════════════════════════════════════════════════════════════════════════════
// @Roles() decorator
//
// Attaches the set of roles allowed to access a route (or controller). Read by
// AdminGuard via Reflector. When absent, AdminGuard defaults to requiring the
// 'admin' role — so simply stacking AdminGuard already locks a route to admins.
// ══════════════════════════════════════════════════════════════════════════════

import { SetMetadata } from '@nestjs/common';
import { UserRole } from '../../schemas/user.schema';

export const ROLES_KEY = 'roles';

export const Roles = (...roles: UserRole[]) => SetMetadata(ROLES_KEY, roles);
