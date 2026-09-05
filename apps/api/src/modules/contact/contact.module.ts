import { Module } from '@nestjs/common';
import { AuthModule } from '../auth/auth.module';
import { UsersModule } from '../users/users.module';
import { GatewayModule } from '../gateway/gateway.module';
import { NotificationsModule } from '../notifications/notifications.module';
import { ContactService } from './contact.service';
import { ContactReminderService } from './contact-reminder.service';
import { ContactController } from './contact.controller';

@Module({
  // UsersModule → consumeBid-free UsersService (findByIdOrNull, applyRating).
  // GatewayModule → WS rooms. NotificationsModule → FCM + inbox sender.
  // ContactInteraction model comes from the @Global DatabaseModule.
  // None depend on ContactService, so there is no cycle.
  imports: [AuthModule, UsersModule, GatewayModule, NotificationsModule],
  controllers: [ContactController],
  providers: [ContactService, ContactReminderService],
  exports: [ContactService],
})
export class ContactModule {}
