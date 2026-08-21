// apps/api/src/modules/media/media.controller.ts
//
// ARCHITECTURE — Cloudinary (CDN public)
//
// Routes :
//   POST   /media/upload/image     → upload image (auth requis) → UploadResult
//   POST   /media/upload/portfolio → photo de réalisation (dossier portfolio/)
//   POST   /media/upload/video     → upload vidéo (auth requis) → UploadResult
//   POST   /media/upload/audio     → upload audio (auth requis) → UploadResult
//   DELETE /media/object/*         → suppression par storedPath (auth requis)
//   DELETE /media/:bucket/:key     → legacy, voir limitation dans MediaService
//
// MIGRATION v15 — MinIO → Cloudinary :
//   La route GET /media/object/* (proxy de streaming MinIO) est SUPPRIMÉE.
//   Cloudinary fournit déjà une URL CDN publique et permanente (`url` dans
//   UploadResult) — il n'y a plus besoin de proxy applicatif.
//   ⚠️ Côté Flutter : afficher directement `url` (persisté en base), ne plus
//   appeler GET /media/object/* ni reconstruire via MediaPathHelper.toUrl().

import {
  BadRequestException,
  Controller,
  Delete,
  HttpCode,
  HttpStatus,
  Post,
  Req,
  Param,
  UploadedFile,
  UseGuards,
  UseInterceptors,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { Request } from 'express';
import { FirebaseAuthGuard, SkipApprovalGate } from '../../common/guards/firebase-auth.guard';
import { CurrentUser }       from '../../common/decorators/current-user.decorator';
import { AuthUser }          from '../../common/guards/firebase-auth.guard';
import { MediaService, UploadResult } from './media.service';

@Controller('media')
export class MediaController {
  constructor(private readonly mediaService: MediaService) {}

  // ══════════════════════════════════════════════════════════════════════════
  // AUTHENTICATED UPLOADS
  // ══════════════════════════════════════════════════════════════════════════

  /**
   * POST /media/upload/image
   * Body: multipart/form-data, champ "file" (JPEG / PNG / WebP, max 10 MB)
   * IMPORTANT : persister `url` dans MongoDB (permanent avec Cloudinary).
   */
  @Post('upload/image')
  @HttpCode(HttpStatus.OK)
  @UseGuards(FirebaseAuthGuard)
  @UseInterceptors(FileInterceptor('file', { limits: { fileSize: 10 * 1024 * 1024 } }))
  async uploadImage(
    @UploadedFile() file: Express.Multer.File | undefined,
    @CurrentUser() user: AuthUser,
  ): Promise<UploadResult> {
    if (!file?.buffer?.length) throw new BadRequestException('file is required');
    return this.mediaService.uploadImage(file.buffer, file.mimetype, user.uid);
  }

  /**
   * POST /media/upload/document
   * Body: multipart/form-data, champ "file" (PDF / JPEG / PNG / WebP, max 10 MB)
   * Documents de vérification (dossier légal worker/business).
   * @SkipApprovalGate : un compte rejeté doit pouvoir re-téléverser ses
   * documents pour re-soumettre — c'est précisément l'état que le gate bloque.
   */
  @Post('upload/document')
  @HttpCode(HttpStatus.OK)
  @UseGuards(FirebaseAuthGuard)
  @SkipApprovalGate()
  @UseInterceptors(FileInterceptor('file', { limits: { fileSize: 10 * 1024 * 1024 } }))
  async uploadDocument(
    @UploadedFile() file: Express.Multer.File | undefined,
    @CurrentUser() user: AuthUser,
  ): Promise<UploadResult> {
    if (!file?.buffer?.length) throw new BadRequestException('file is required');
    return this.mediaService.uploadDocument(file.buffer, file.mimetype, user.uid);
  }

  /**
   * POST /media/upload/portfolio
   * Body: multipart/form-data, champ "file" (JPEG / PNG / WebP, max 10 MB)
   * Photo de réalisation (galerie publique du worker) → dossier portfolio/.
   * L'URL retournée est ensuite persistée via PUT /workers/:id/portfolio.
   */
  @Post('upload/portfolio')
  @HttpCode(HttpStatus.OK)
  @UseGuards(FirebaseAuthGuard)
  @UseInterceptors(FileInterceptor('file', { limits: { fileSize: 10 * 1024 * 1024 } }))
  async uploadPortfolioImage(
    @UploadedFile() file: Express.Multer.File | undefined,
    @CurrentUser() user: AuthUser,
  ): Promise<UploadResult> {
    if (!file?.buffer?.length) throw new BadRequestException('file is required');
    return this.mediaService.uploadPortfolioImage(file.buffer, file.mimetype, user.uid);
  }

  /** POST /media/upload/video — multipart, champ "file" (max 100 MB) */
  @Post('upload/video')
  @HttpCode(HttpStatus.OK)
  @UseGuards(FirebaseAuthGuard)
  @UseInterceptors(FileInterceptor('file', { limits: { fileSize: 100 * 1024 * 1024 } }))
  async uploadVideo(
    @UploadedFile() file: Express.Multer.File | undefined,
    @CurrentUser() user: AuthUser,
  ): Promise<UploadResult> {
    if (!file?.buffer?.length) throw new BadRequestException('file is required');
    return this.mediaService.uploadVideo(file.buffer, file.mimetype, user.uid);
  }

  /** POST /media/upload/audio — multipart, champ "file" (max 50 MB) */
  @Post('upload/audio')
  @HttpCode(HttpStatus.OK)
  @UseGuards(FirebaseAuthGuard)
  @UseInterceptors(FileInterceptor('file', { limits: { fileSize: 50 * 1024 * 1024 } }))
  async uploadAudio(
    @UploadedFile() file: Express.Multer.File | undefined,
    @CurrentUser() user: AuthUser,
  ): Promise<UploadResult> {
    if (!file?.buffer?.length) throw new BadRequestException('file is required');
    return this.mediaService.uploadAudio(file.buffer, file.mimetype, user.uid);
  }

  // ══════════════════════════════════════════════════════════════════════════
  // AUTHENTICATED DELETES
  // ══════════════════════════════════════════════════════════════════════════

  /**
   * DELETE /media/object/* — suppression par storedPath. ENDPOINT PRÉFÉRÉ.
   */
  @Delete('object/*')
  @HttpCode(HttpStatus.NO_CONTENT)
  @UseGuards(FirebaseAuthGuard)
  async deleteByPath(
    @Req()         req: Request,
    @CurrentUser() user: AuthUser,
  ): Promise<void> {
    const storedPath = (req.params as Record<string, string>)['0'] ?? '';
    if (!storedPath) throw new BadRequestException('Missing media path');
    return this.mediaService.deleteByStoredPath(storedPath, user.uid);
  }

  /**
   * @deprecated Voir limitation détaillée dans MediaService.deleteFile.
   * Utiliser DELETE /media/object/* à la place.
   */
  @Delete(':bucket/:key')
  @HttpCode(HttpStatus.NO_CONTENT)
  @UseGuards(FirebaseAuthGuard)
  async deleteFile(
    @Param('bucket') bucket: string,
    @Param('key')    key: string,
    @CurrentUser()   user: AuthUser,
  ): Promise<void> {
    return this.mediaService.deleteFile(bucket, decodeURIComponent(key), user.uid);
  }
}
