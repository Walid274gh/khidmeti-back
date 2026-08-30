import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { HfQueueService } from './hf-queue.service';
const VISION_TIMEOUT_MS = 20_000;
@Injectable()
export class VisionService {
  private readonly logger = new Logger(VisionService.name);
  private readonly url?: string;
  constructor(private readonly config: ConfigService, private readonly hf: HfQueueService) { this.url = this.config.get<string>('VISION_URL') || undefined; }
  async classify(image: Buffer): Promise<{ profession: string; confidence: number } | null> {
    if (this.hf.enabled) {
      try {
        const base = this.config.get<string>('HF_SPACE_URL')!.replace(/\/+$/,'');
        const blob = new Blob([new Uint8Array(image)], { type: 'image/jpeg' });
        const fd = new FormData(); fd.append('files', blob as unknown as Blob, 'image.jpg');
        const up = await fetch(`${base}/upload`, { method: 'POST', body: fd as unknown as BodyInit, signal: AbortSignal.timeout(15_000) });
        if (up.ok) {
          const paths = await up.json() as string[];
          const raw = await this.hf.call(1, [{ path: paths[0], url: `${base}/file=${paths[0]}`, orig_name: 'image.jpg', size: image.length, mime_type: 'image/jpeg' }]);
          const body = JSON.parse(raw) as { profession?: unknown; profession_confidence?: unknown };
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
