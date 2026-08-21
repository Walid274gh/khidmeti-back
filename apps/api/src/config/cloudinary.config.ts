import { Injectable, OnModuleInit } from '@nestjs/common';
import { v2 as cloudinary } from 'cloudinary';

@Injectable()
export class CloudinaryConfigService implements OnModuleInit {
  onModuleInit(): void {
    cloudinary.config({
      cloud_name: process.env['CLOUDINARY_CLOUD_NAME'] ?? '',
      api_key:    process.env['CLOUDINARY_API_KEY'] ?? '',
      api_secret: process.env['CLOUDINARY_API_SECRET'] ?? '',
      secure:     true,
    });
  }

  get client() {
    return cloudinary;
  }

  /** Our account id — the first path segment of every delivery URL we issue. */
  get cloudName(): string {
    return process.env['CLOUDINARY_CLOUD_NAME'] ?? '';
  }

  get folderProfiles(): string {
    return process.env['CLOUDINARY_FOLDER_PROFILES'] ?? 'profile-images';
  }

  get folderMedia(): string {
    return process.env['CLOUDINARY_FOLDER_MEDIA'] ?? 'service-media';
  }

  get folderAudio(): string {
    return process.env['CLOUDINARY_FOLDER_AUDIO'] ?? 'audio-recordings';
  }

  get folderDocuments(): string {
    return process.env['CLOUDINARY_FOLDER_DOCUMENTS'] ?? 'verification-docs';
  }

  /**
   * Worker showcase photos. Kept out of folderProfiles so the public gallery
   * can be listed, quota-audited or purged per worker without touching avatars.
   */
  get folderPortfolio(): string {
    return process.env['CLOUDINARY_FOLDER_PORTFOLIO'] ?? 'portfolio';
  }
}
