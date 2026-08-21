// apps/api/src/modules/media/media.service.ts
//
// ARCHITECTURE — Cloudinary (CDN public direct)
//
// MIGRATION v15 : MinIO (privé, proxy NestJS) → Cloudinary (CDN public)
//
//   AVANT (MinIO) :
//     Flutter → NestJS proxy (/media/object/*) → MinIO privé (réseau Docker)
//     storedPath = "bucket/userId/file.ext" stocké en base, URL reconstruite
//     dynamiquement (dépendante du domaine API / tunnel Cloudflare).
//
//   APRÈS (Cloudinary) :
//     Flutter → URL Cloudinary directe (CDN, publique, permanente)
//     Plus de proxy nécessaire — Cloudinary EST déjà un CDN public.
//
//   ⚠️ CÔTÉ FLUTTER : persister `url` (permanent) au lieu de reconstruire
//   via MediaPathHelper.toUrl(storedPath, ...). `storedPath` ne sert plus
//   qu'à la suppression (DELETE), pas à l'affichage.
//
// INTERFACE UploadResult (forme inchangée — compat Flutter) :
//   • url        : URL Cloudinary complète et permanente → PERSISTER en base
//   • key        : identifiant court du fichier ("timestamp_uuid", sans extension)
//   • storedPath : "resourceType/folder/userId/key" → pour DELETE uniquement

import { BadRequestException, Injectable, Logger } from '@nestjs/common';
import { randomUUID } from 'crypto';
import type { UploadApiResponse, UploadApiErrorResponse } from 'cloudinary';
import { CloudinaryConfigService } from '../../config/cloudinary.config';

// ── Public interface ──────────────────────────────────────────────────────────

export interface UploadResult {
  /** URL Cloudinary complète et permanente. PERSISTER CECI en base. */
  url: string;
  /** Identifiant court du fichier. Ex: "1719312345678_a1b2c3d4-..." */
  key: string;
  /** "resourceType/folder/userId/key" — utilisé uniquement pour la suppression. */
  storedPath: string;
}

type CloudinaryResourceType = 'image' | 'video' | 'raw';

/**
 * Reverse of `upload()`: parse a delivery URL we issued back into the pieces
 * needed to identify and destroy it. Returns null for anything that is not a
 * Cloudinary URL of ours (shape: /<cloud>/<type>/upload/v<n>/<publicId>.<ext>).
 *
 * Used to validate client-supplied media URLs before persisting them on a
 * public profile — an unvalidated URL field is an open redirect / image-host
 * injection into every client's app.
 */
export function parseCloudinaryUrl(
  url: string,
): { cloudName: string; resourceType: CloudinaryResourceType; publicId: string } | null {
  const m = /^https:\/\/res\.cloudinary\.com\/([^/]+)\/(image|video|raw)\/upload\/v\d+\/(.+)$/
    .exec(url ?? '');
  if (!m) return null;
  const resourceType = m[2] as CloudinaryResourceType;
  // Cloudinary appends the delivery extension to image/video public_ids, and
  // `destroy` expects it stripped. 'raw' is the opposite: upload() bakes the
  // extension INTO the public_id (see the PDF path), so it must stay.
  const publicId =
    resourceType === 'raw' ? m[3] : m[3].replace(/\.[a-z0-9]{2,5}$/i, '');
  return { cloudName: m[1], resourceType, publicId };
}

// ── Service ───────────────────────────────────────────────────────────────────

@Injectable()
export class MediaService {
  private readonly logger = new Logger(MediaService.name);

  constructor(private readonly config: CloudinaryConfigService) {}

  // ── Upload methods ────────────────────────────────────────────────────────

  async uploadImage(buffer: Buffer, _mime: string, userId: string): Promise<UploadResult> {
    this.validateImageMagicBytes(buffer);
    if (buffer.length > 10 * 1024 * 1024) {
      throw new BadRequestException('Image size exceeds 10 MB limit');
    }
    return this.upload(buffer, 'image', this.config.folderProfiles, userId);
  }

  /**
   * Worker showcase photo — same validation as a profile image, its own folder
   * (`portfolio/<uid>/…`) so gallery assets stay separable from avatars for
   * auditing and cleanup. isOwnPortfolioImage() also keys on that prefix.
   */
  async uploadPortfolioImage(buffer: Buffer, _mime: string, userId: string): Promise<UploadResult> {
    this.validateImageMagicBytes(buffer);
    if (buffer.length > 10 * 1024 * 1024) {
      throw new BadRequestException('Image size exceeds 10 MB limit');
    }
    return this.upload(buffer, 'image', this.config.folderPortfolio, userId);
  }

  async uploadVideo(buffer: Buffer, _mime: string, userId: string): Promise<UploadResult> {
    if (buffer.length > 100 * 1024 * 1024) {
      throw new BadRequestException('Video size exceeds 100 MB limit');
    }
    return this.upload(buffer, 'video', this.config.folderMedia, userId);
  }

  async uploadAudio(buffer: Buffer, _mime: string, userId: string): Promise<UploadResult> {
    if (buffer.length > 50 * 1024 * 1024) {
      throw new BadRequestException('Audio size exceeds 50 MB limit');
    }
    // Cloudinary n'a pas de resource_type "audio" dédié : les fichiers
    // audio-only (m4a/mp3/wav…) passent par le pipeline "video" — c'est le
    // comportement documenté et attendu de l'API Cloudinary.
    return this.upload(buffer, 'video', this.config.folderAudio, userId);
  }

  /**
   * Document de vérification (dossier légal worker/business) : image OU PDF,
   * max 10 MB. Les images passent par le pipeline image ; les PDF partent en
   * resource_type 'raw' (fichier original servi tel quel — la livraison PDF
   * via 'image' est désactivée par défaut sur les comptes Cloudinary récents).
   */
  async uploadDocument(buffer: Buffer, _mime: string, userId: string): Promise<UploadResult> {
    if (buffer.length > 10 * 1024 * 1024) {
      throw new BadRequestException('Document size exceeds 10 MB limit');
    }
    const isPdf =
      buffer.length >= 4 &&
      buffer[0] === 0x25 && buffer[1] === 0x50 && // %P
      buffer[2] === 0x44 && buffer[3] === 0x46;   // DF
    if (isPdf) {
      return this.upload(buffer, 'raw', this.config.folderDocuments, userId, '.pdf');
    }
    this.validateImageMagicBytes(buffer); // throws unless JPEG/PNG/WebP
    return this.upload(buffer, 'image', this.config.folderDocuments, userId);
  }

  // ── Ownership ─────────────────────────────────────────────────────────────

  /**
   * True when [url] is a portfolio IMAGE on our own Cloudinary account,
   * uploaded by [userId] — `upload()` puts folder and uploader id in the
   * public_id path, so the URL itself carries the proof. Guards the portfolio
   * endpoint, which persists a client-supplied URL instead of the bytes.
   *
   * The folder is part of the check: an avatar or a verification document must
   * not be re-listed as a showcase photo, or the takedown/quota story breaks
   * (and a private ID scan could be republished as public gallery content).
   */
  isOwnPortfolioImage(url: string, userId: string): boolean {
    const parsed = parseCloudinaryUrl(url);
    return (
      parsed !== null &&
      parsed.resourceType === 'image' &&
      parsed.cloudName === this.config.cloudName &&
      parsed.publicId.startsWith(`${this.config.folderPortfolio}/${userId}/`)
    );
  }

  /**
   * Best-effort cleanup of an asset we no longer reference (portfolio photo
   * removed). Never throws — a dangling Cloudinary object must not fail the
   * user-facing write that dropped it.
   */
  async destroyByUrl(url: string): Promise<void> {
    const parsed = parseCloudinaryUrl(url);
    if (!parsed) return;
    try {
      await this.destroy(parsed.publicId, parsed.resourceType);
    } catch {
      // already logged in destroy()
    }
  }

  // ── Delete methods ────────────────────────────────────────────────────────

  /**
   * Supprime par storedPath ("resourceType/folder/userId/key").
   * Vérifie l'ownership : userId doit correspondre au 3ème segment.
   */
  async deleteByStoredPath(storedPath: string, userId: string): Promise<void> {
    const parts = storedPath.split('/');
    if (parts.length < 4) {
      throw new BadRequestException('Invalid stored path format');
    }

    const [resourceType, folder, ownerId, ...rest] = parts;
    if (ownerId !== userId) {
      throw new BadRequestException(
        'Ownership check failed: key does not belong to this user',
      );
    }
    if (resourceType !== 'image' && resourceType !== 'video' && resourceType !== 'raw') {
      throw new BadRequestException('Invalid resource type in stored path');
    }

    const publicId = `${folder}/${ownerId}/${rest.join('/')}`;
    await this.destroy(publicId, resourceType);
  }

  /**
   * @deprecated DELETE /media/:bucket/:key (legacy).
   * ⚠️ LIMITATION avec Cloudinary : Express ne capture pas les "/" dans un
   * paramètre de route simple (:key), or les publicId Cloudinary contiennent
   * "folder/userId/filename" (2 slashes). Cet endpoint ne peut donc PAS
   * fonctionner pour des médias uploadés après cette migration — utiliser
   * exclusivement DELETE /media/object/* (deleteByStoredPath) désormais.
   * Conservé uniquement pour ne pas casser un éventuel appel existant côté
   * client (il échouera proprement avec une 400, pas un crash serveur).
   */
  async deleteFile(resourceTypeRaw: string, key: string, userId: string): Promise<void> {
    if (resourceTypeRaw !== 'image' && resourceTypeRaw !== 'video') {
      throw new BadRequestException(
        'Invalid resource type — use DELETE /media/object/* for Cloudinary media',
      );
    }
    const segments = key.split('/');
    const ownerId  = segments[1]; // attendu: folder/userId/filename
    if (ownerId !== userId) {
      throw new BadRequestException(
        'Ownership check failed: key does not belong to this user',
      );
    }
    await this.destroy(key, resourceTypeRaw);
  }

  // ── Private helpers ───────────────────────────────────────────────────────

  private async upload(
    buffer: Buffer,
    resourceType: CloudinaryResourceType,
    folder: string,
    userId: string,
    // Pour resource_type 'raw', Cloudinary sert le fichier à l'URL exacte du
    // public_id — l'extension doit donc en faire partie (".pdf") pour que le
    // navigateur/l'app reconnaisse le type. Vide pour image/video.
    extension = '',
  ): Promise<UploadResult> {
    const key      = `${Date.now()}_${randomUUID()}`;
    const publicId = `${folder}/${userId}/${key}${extension}`;

    const result = await new Promise<UploadApiResponse>((resolve, reject) => {
      const stream = this.config.client.uploader.upload_stream(
        {
          public_id:     publicId,
          resource_type: resourceType,
          overwrite:     false,
          timeout:       60_000, // a hung Cloudinary socket must not pin the buffer forever
        },
        (error?: UploadApiErrorResponse, res?: UploadApiResponse) => {
          if (error || !res) {
            return reject(error ?? new Error('Cloudinary upload failed: no result'));
          }
          resolve(res);
        },
      );
      stream.end(buffer);
    }).catch((err) => {
      this.logger.error(`Cloudinary upload failed: ${publicId}`, err);
      throw err;
    });

    const storedPath = `${resourceType}/${publicId}`;
    return { url: result.secure_url, key, storedPath };
  }

  private async destroy(
    publicId: string,
    resourceType: CloudinaryResourceType,
  ): Promise<void> {
    try {
      const res = await this.config.client.uploader.destroy(publicId, {
        resource_type: resourceType,
      });
      if (res.result !== 'ok' && res.result !== 'not found') {
        this.logger.warn(
          `Cloudinary destroy unexpected result for "${publicId}": ${res.result}`,
        );
      }
    } catch (err) {
      this.logger.error(`MediaService.destroy failed: ${publicId}`, err);
      throw err;
    }
  }

  private validateImageMagicBytes(buffer: Buffer): void {
    if (buffer.length < 4) throw new BadRequestException('File too small');

    const isJpeg = buffer[0] === 0xff && buffer[1] === 0xd8 && buffer[2] === 0xff;
    const isPng  =
      buffer[0] === 0x89 && buffer[1] === 0x50 &&
      buffer[2] === 0x4e && buffer[3] === 0x47;
    const isWebp =
      buffer.length >= 12 &&
      buffer[0] === 0x52 && buffer[1] === 0x49 &&
      buffer[2] === 0x46 && buffer[3] === 0x46 &&
      buffer[8] === 0x57 && buffer[9] === 0x45 &&
      buffer[10] === 0x42 && buffer[11] === 0x50;

    if (!isJpeg && !isPng && !isWebp) {
      throw new BadRequestException(
        'Invalid image format. Only JPEG, PNG, and WebP are allowed.',
      );
    }
  }
}
