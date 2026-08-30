// vision.service — HF Gradio queue priority, fallback local ai-vision
import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { HfQueueService } from './hf-queue.service';
const VISION_TIMEOUT_MS = 20_000;
const VISION_BASE = 'https://walid274gh-khidmeti-ai.hf.space';
export interface VisionResult { profession: string; confidence: number; }
@Injectable()
export class VisionService {
  private readonly logger = new Logger(VisionService.name);
  private readonly url?: string;
  private warnedDisabled = false;
  constructor(private readonly config: ConfigService, private readonly hf: HfQueueService) {
    this.url = this.config.get<string>('VISION_URL') || undefined;
  }
  get enabled(): boolean { return this.url != null || this.hf.enabled; }
  async classify(image: Buffer): Promise<VisionResult | null> {
    if (this.hf.enabled) {
      try {
        const HF_URL = this.config.get<string>('HF_SPACE_URL') || VISION_BASE;
        const blob = new Blob([new Uint8Array(image)], { type: 'image/jpeg' });
        const fd = new FormData(); fd.append('files', blob, 'image.jpg');
        const up = await fetch(`${HF_URL.replace(/\/+$/,'')}/upload`, { method: 'POST', body: fd as unknown as BodyInit, signal: AbortSignal.timeout(15_000) });
        if (up.ok) {
          const paths = await up.json() as string[];
          const raw = await this.hf.call(1, [{ path: paths[0], url: `${HF_URL}/file=${paths[0]}`, orig_name: 'image.jpg', size: image.length, mime_type: 'image/jpeg' }]);
          const body = JSON.parse(raw) as { profession?: unknown; profession_confidence?: unknown };
          if (typeof body.profession === 'string') return { profession: body.profession, confidence: Number(body.profession_confidence) || 0 };
        }
      } catch (err) { this.logger.warn(`HF vision queue: ${(err as Error).message}`); }
      return null;
    }
    if (!this.url) { if (!this.warnedDisabled) { this.warnedDisabled = true; this.logger.warn('VISION_URL/HF_SPACE_URL missing — vision disabled'); } return null; }
    try {
      const res = await fetch(`${this.url.replace(/\/+$/,'')}/classify`, { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: new Uint8Array(image), signal: AbortSignal.timeout(VISION_TIMEOUT_MS) });
      if (!res.ok) { this.logger.warn(`ai-vision HTTP ${res.status}`); return null; }
      const body = await res.json() as { profession?: unknown; profession_confidence?: unknown };
      if (typeof body.profession !== 'string') { this.logger.warn('ai-vision malformed'); return null; }
      return { profession: body.profession, confidence: Number(body.profession_confidence) || 0 };
    } catch (err) { this.logger.warn(`ai-vision: ${(err as Error).message}`); return null; }
  }
}
