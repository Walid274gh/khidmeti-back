import { Module } from '@nestjs/common';
import { MediaService } from './media.service';
import { MediaController } from './media.controller';
import { CloudinaryConfigService } from '../../config/cloudinary.config';
import { AuthModule } from '../auth/auth.module';

@Module({
  imports: [AuthModule],
  controllers: [MediaController],
  providers: [MediaService, CloudinaryConfigService],
  exports: [MediaService],
})
export class MediaModule {}
