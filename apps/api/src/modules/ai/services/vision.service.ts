import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
const VISION_TIMEOUT_MS = 20_000;
export interface VisionResult { profession: string; confidence: number; }
@Injectable()
export class VisionService {
  private readonly logger = new Logger(VisionService.name);
  private readonly hfUrl?: string;
  private readonly url?: string;
  constructor(private readonly config: ConfigService) {
    this.hfUrl = this.config.get<string>('HF_SPACE_URL')?.replace(/\/+$/, '');
    this.url = this.config.get<string>('VISION_URL') || undefined;
  }
  get enabled(): boolean { return !!(this.hfUrl || this.url); }
  async classify(image: Buffer): Promise<VisionResult | null> {
    if (this.hfUrl) {
      try {
        const blob = new Blob([new Uint8Array(image)], { type: 'image/jpeg' });
        const fd = new FormData(); fd.append('file', blob, 'image.jpg');
        const res = await fetch(`${this.hfUrl}/vision`, { method: 'POST', body: fd as unknown as BodyInit, signal: AbortSignal.timeout(VISION_TIMEOUT_MS) });
        if (res.ok) {
          const body = await res.json() as { profession?: unknown; profession_confidence?: unknown };
          if (typeof body.profession === 'string') return { profession: body.profession, confidence: Number(body.profession_confidence)||0 };
        }
      } catch (e) { this.logger.warn(`HF vision: ${(e as Error).message}`); }
    }
    if (!this.url) return null;
    try {
      const res = await fetch(`${this.url.replace(/\/+$/,'')}/classify`, { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: new Uint8Array(image), signal: AbortSignal.timeout(VISION_TIMEOUT_MS) });
      if (!res.ok) return null;
      const body = await res.json() as { profession?: unknown; profession_confidence?: unknown };
      if (typeof body.profession !== 'string') return null;
      return { profession: body.profession, confidence: Number(body.profession_confidence)||0 };
    } catch (e) { this.logger.warn(`vision: ${(e as Error).message}`); return null; }
  }
}
