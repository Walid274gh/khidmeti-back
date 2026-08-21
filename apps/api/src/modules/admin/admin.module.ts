// ══════════════════════════════════════════════════════════════════════════════
// AdminModule
//
// Wires the admin dashboard API. Models come from the @Global DatabaseModule,
// so no forFeature is needed here. NotificationsModule is imported to reuse
// PushSenderService for broadcasts. AuthModule provides FirebaseAuthGuard;
// AdminGuard is provided locally (it injects UserModel + Reflector).
// ══════════════════════════════════════════════════════════════════════════════

import { Module } from '@nestjs/common';
import { AuthModule } from '../auth/auth.module';
import { NotificationsModule } from '../notifications/notifications.module';
import { AdminController } from './admin.controller';
import { AdminService } from './admin.service';
import { AdminGuard } from '../../common/guards/admin.guard';

@Module({
  imports: [AuthModule, NotificationsModule],
  controllers: [AdminController],
  providers: [AdminService, AdminGuard],
})
export class AdminModule {}
