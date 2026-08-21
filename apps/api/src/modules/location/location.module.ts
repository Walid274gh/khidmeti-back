import { Module } from '@nestjs/common';
import { AuthModule } from '../auth/auth.module';
import { LocationService } from './location.service';
import { LocationController } from './location.controller';

@Module({
  imports: [AuthModule],
  controllers: [LocationController],
  providers: [LocationService],
  exports: [LocationService],
})
export class LocationModule {}
