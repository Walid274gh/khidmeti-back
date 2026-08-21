import { Module } from '@nestjs/common';
import { AuthModule } from '../auth/auth.module';
import { GatewayModule } from '../gateway/gateway.module';
import { NotificationsModule } from '../notifications/notifications.module';
import { UsersModule } from '../users/users.module';
import { BidsService } from './bids.service';
import { BidsController } from './bids.controller';

@Module({
  // GatewayModule exports the WS gateways; NotificationsModule exports the FCM
  // + inbox sender. UsersModule exports UsersService (subscription check).
  // None depend on BidsService, so there is no cycle.
  imports: [AuthModule, GatewayModule, NotificationsModule, UsersModule],
  controllers: [BidsController],
  providers: [BidsService],
  exports: [BidsService],
})
export class BidsModule {}
