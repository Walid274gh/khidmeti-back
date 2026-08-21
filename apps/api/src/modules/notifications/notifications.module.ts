import { Module } from '@nestjs/common';
import { AuthModule } from '../auth/auth.module';
import { NotificationsService } from './notifications.service';
import { NotificationsController } from './notifications.controller';
import { PushSenderService } from './push-sender.service';

@Module({
  imports: [AuthModule],
  controllers: [NotificationsController],
  // User model is available via the @Global DatabaseModule, so PushSenderService
  // can @InjectModel(User.name) without a local forFeature.
  providers: [NotificationsService, PushSenderService],
  exports: [NotificationsService, PushSenderService],
})
export class NotificationsModule {}
