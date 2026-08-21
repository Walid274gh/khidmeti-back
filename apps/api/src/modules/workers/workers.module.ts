import { Module } from '@nestjs/common';
import { AuthModule }       from '../auth/auth.module';
import { UsersModule }      from '../users/users.module';
import { MediaModule }      from '../media/media.module';
import { WorkersService }   from './workers.service';
import { WorkersController } from './workers.controller';

@Module({
  // UsersModule provides UsersService; MediaModule provides MediaService,
  // used to prove portfolio URLs are our own media before they go public.
  imports: [AuthModule, UsersModule, MediaModule],
  controllers: [WorkersController],
  providers:   [WorkersService],
  exports:     [WorkersService],
})
export class WorkersModule {}
