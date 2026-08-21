// apps/api/src/modules/ai/ai.controller.ts
//
// v15 — IA 100% maison : texte = ai-nlu, audio = ai-stt, image = ai-vision.
//
// Le controller reste simple : il reçoit les fichiers (images ou audio) et les
// passe au service.
// Formats : images JPEG/PNG/WebP, audio WAV/MP3/M4A/OGG (PyAV côté ai-stt).
import {
  Controller,
  Post,
  Body,
  UseGuards,
  UseInterceptors,
  UploadedFile,
  BadRequestException,
  HttpCode,
  HttpStatus,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { FirebaseAuthGuard } from '../../common/guards/firebase-auth.guard';
import { CurrentUser } from '../../common/decorators/current-user.decorator';
import type { AuthUser } from '../../common/guards/firebase-auth.guard';
import { IntentExtractorService } from './services/intent-extractor.service';
import type { SearchIntent } from './services/intent-extractor.service';
import { FlywheelService } from './services/flywheel.service';
import { ExtractIntentDto } from './dto/extract-intent.dto';

@Controller('ai')
@UseGuards(FirebaseAuthGuard)
export class AiController {
  constructor(
    private readonly intentExtractor: IntentExtractorService,
    private readonly flywheel: FlywheelService,
  ) {}

  /**
   * POST /ai/extract-intent
   * Texte en Darija / Français / Arabe / mix → intention JSON
   */
  @Post('extract-intent')
  @HttpCode(HttpStatus.OK)
  async extractIntent(
    @Body() dto: ExtractIntentDto,
    @CurrentUser() user: AuthUser,
  ): Promise<SearchIntent> {
    const result = await this.intentExtractor.extractFromText(dto.text, user.uid);
    // C1 flywheel — consent-gated, fire-and-forget, never blocks the response.
    this.flywheel.log(user.uid, 'text', dto.text, result);
    return result;
  }

  /**
   * POST /ai/extract-intent/audio
   * Audio (m4a / wav / mp3 / ogg) → faster-whisper STT → dziriBERT intent
   * Limite : 50 MB
   */
  @Post('extract-intent/audio')
  @HttpCode(HttpStatus.OK)
  @UseInterceptors(FileInterceptor('file', { limits: { fileSize: 50 * 1024 * 1024 } }))
  async extractIntentFromAudio(
    @UploadedFile() file: Express.Multer.File | undefined,
    @CurrentUser() user: AuthUser,
  ): Promise<SearchIntent> {
    if (!file?.buffer?.length) {
      throw new BadRequestException('Fichier audio requis (m4a, wav, mp3, ogg)');
    }
    const result = await this.intentExtractor.extractFromAudio(file.buffer, file.mimetype, user.uid);
    // C1 flywheel — audio bytes + transcription = future Whisper LoRA pairs.
    this.flywheel.log(
      user.uid, 'audio', result.transcribedText ?? '', result,
      file.buffer, (file.mimetype.split('/')[1] ?? 'bin'),
    );
    return result;
  }

  /**
   * POST /ai/extract-intent/image
   * Image (JPEG, PNG, WebP) → SigLIP2 zero-shot (ai-vision) → intent JSON
   * Limite : 10 MB
   *
   * Formats supportés (détection par magic bytes) :
   *   JPEG : FF D8 FF
   *   PNG  : 89 50 4E 47
   *   WebP : RIFF....WEBP (commun sur Android)
   */
  @Post('extract-intent/image')
  @HttpCode(HttpStatus.OK)
  @UseInterceptors(FileInterceptor('file', { limits: { fileSize: 10 * 1024 * 1024 } }))
  async extractIntentFromImage(
    @UploadedFile() file: Express.Multer.File | undefined,
    @CurrentUser() user: AuthUser,
  ): Promise<SearchIntent> {
    if (!file?.buffer?.length) {
      throw new BadRequestException('Fichier image requis (JPEG, PNG ou WebP)');
    }

    const b = file.buffer;

    // Détection par magic bytes — plus fiable que le Content-Type header
    const isJpeg = b.length >= 3  && b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff;
    const isPng  = b.length >= 4  && b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47;
    const isWebp = b.length >= 12 &&
      b[0] === 0x52 && b[1] === 0x49 && b[2] === 0x46 && b[3] === 0x46 && // RIFF
      b[8] === 0x57 && b[9] === 0x45 && b[10] === 0x42 && b[11] === 0x50;  // WEBP

    if (!isJpeg && !isPng && !isWebp) {
      throw new BadRequestException(
        'Format image non supporté. Formats acceptés : JPEG, PNG, WebP',
      );
    }

    const result = await this.intentExtractor.extractFromImage(b.toString('base64'), user.uid);
    // C1 flywheel — the photo is the training input; ext from magic bytes.
    this.flywheel.log(
      user.uid, 'image', '', result,
      b, isJpeg ? 'jpg' : isPng ? 'png' : 'webp',
    );
    return result;
  }
}
