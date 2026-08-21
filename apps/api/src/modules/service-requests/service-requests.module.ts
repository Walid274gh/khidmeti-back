import { Module } from '@nestjs/common';
import { AuthModule }                from '../auth/auth.module';
import { UsersModule }               from '../users/users.module';
import { GatewayModule }             from '../gateway/gateway.module';
import { NotificationsModule }       from '../notifications/notifications.module';
import { AiModule }                  from '../ai/ai.module';
import { ServiceRequestsService }    from './service-requests.service';
import { ServiceRequestsController } from './service-requests.controller';

@Module({
  // GatewayModule → ServiceRequestGateway (WS); NotificationsModule → FCM+inbox.
  // AiModule → FlywheelService (C1: request creation labels the AI query).
  imports: [AuthModule, UsersModule, GatewayModule, NotificationsModule, AiModule],
  controllers: [ServiceRequestsController],
  providers:   [ServiceRequestsService],
  exports:     [ServiceRequestsService],
})
export class ServiceRequestsModule {}
