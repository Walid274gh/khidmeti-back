import { Module } from '@nestjs/common';
import { AuthModule }    from '../auth/auth.module';
import { NotificationsModule } from '../notifications/notifications.module';
import { UsersService }  from './users.service';
import { UsersController } from './users.controller';

@Module({
  // NotificationsModule exports PushSenderService — used to alert admins when a
  // worker/business submits verification documents.
  imports: [AuthModule, NotificationsModule],
  controllers: [UsersController],
  providers: [UsersService],
  exports: [UsersService],   // ← exported so WorkersModule / BidsModule can import it
})
export class UsersModule {}
