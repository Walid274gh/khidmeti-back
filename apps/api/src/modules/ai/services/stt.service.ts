// stt.service — HF Gradio queue priority, fallback local ai-stt
import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { HfQueueService } from './hf-queue.service';
const STT_TIMEOUT_MS = 120_000;
const MAX_STT_BYTES = 16 * 1024 * 1024;
const STT_BASE = 'https://walid274gh-khidmeti-ai.hf.space';
export interface SttResult { text: string; language: string; duration: number; }
@Injectable()
export class SttService {
  private readonly logger = new Logger(SttService.name);
  private readonly url?: string;
  private warnedDisabled = false;
  constructor(private readonly config: ConfigService, private readonly hf: HfQueueService) {
    this.url = this.config.get<string>('STT_URL') || undefined;
  }
  get enabled(): boolean { return this.url != null || this.hf.enabled; }
  async transcribe(buffer: Buffer, mime: string): Promise<SttResult | null> {
    if (buffer.length > MAX_STT_BYTES) { this.logger.warn(`Audio ${(buffer.length/1024/1024).toFixed(1)}MB > limit`); return null; }
    // HF ZeroGPU via Gradio queue (file upload)
    if (this.hf.enabled) {
      try {
        const HF_URL = this.config.get<string>('HF_SPACE_URL') || STT_BASE;
        // Gradio File upload: POST /upload with multipart
        const blob = new Blob([new Uint8Array(buffer)], { type: mime || 'audio/webm' });
        const fd = new FormData(); fd.append('files', blob, 'audio.webm');
        const up = await fetch(`${HF_URL.replace(/\/+$/,'')}/upload`, { method: 'POST', body: fd as unknown as BodyInit, signal: AbortSignal.timeout(15_000) });
        if (up.ok) {
          const paths = await up.json() as string[];
          const raw = await this.hf.call(0, [{ path: paths[0], url: `${HF_URL}/file=${paths[0]}`, orig_name: 'audio.webm', size: buffer.length, mime_type: mime }]);
          const text = raw.trim();
          return { text, language: 'ar', duration: 0 };
        }
      } catch (err) { this.logger.warn(`HF STT queue: ${(err as Error).message}`); }
      return null;
    }
    if (!this.url) { if (!this.warnedDisabled) { this.warnedDisabled = true; this.logger.warn('STT_URL/HF_SPACE_URL missing — STT disabled'); } return null; }
    try {
      const res = await fetch(`${this.url.replace(/\/+$/,'')}/transcribe`, { method: 'POST', headers: { 'Content-Type': mime || 'application/octet-stream' }, body: new Uint8Array(buffer), signal: AbortSignal.timeout(STT_TIMEOUT_MS) });
      if (!res.ok) { this.logger.warn(`ai-stt HTTP ${res.status}`); return null; }
      const body = await res.json() as Partial<SttResult>;
      if (typeof body.text !== 'string') { this.logger.warn('ai-stt malformed'); return null; }
      return { text: body.text, language: String(body.language ?? ''), duration: Number(body.duration) || 0 };
    } catch (err) { this.logger.warn(`ai-stt: ${(err as Error).message}`); return null; }
  }
}
