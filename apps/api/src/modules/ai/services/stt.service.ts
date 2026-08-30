import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { HfQueueService } from './hf-queue.service';
const STT_TIMEOUT_MS = 120_000;
@Injectable()
export class SttService {
  private readonly logger = new Logger(SttService.name);
  private readonly url?: string;
  constructor(private readonly config: ConfigService, private readonly hf: HfQueueService) { this.url = this.config.get<string>('STT_URL') || undefined; }
  async transcribe(buffer: Buffer, mime: string): Promise<{ text: string; language: string; duration: number } | null> {
    if (this.hf.enabled) {
      try {
        const base = this.config.get<string>('HF_SPACE_URL')!.replace(/\/+$/,'');
        const blob = new Blob([new Uint8Array(buffer)], { type: mime || 'audio/webm' });
        const fd = new FormData(); (fd as unknown as Record<string, unknown>).append = (fd.append as unknown as (k:string,v:unknown,f?:string)=>void);
        fd.append('files', blob as unknown as Blob, 'audio.webm');
        const up = await fetch(`${base}/upload`, { method: 'POST', body: fd as unknown as BodyInit, signal: AbortSignal.timeout(15_000) });
        if (up.ok) {
          const paths = await up.json() as string[];
          const raw = await this.hf.call(0, [{ path: paths[0], url: `${base}/file=${paths[0]}`, orig_name: 'audio.webm', size: buffer.length, mime_type: mime }]);
          return { text: raw.trim(), language: 'ar', duration: 0 };
        }
      } catch (e) { this.logger.warn(`HF stt: ${(e as Error).message}`); }
    }
    if (!this.url) return null;
    try {
      const res = await fetch(`${this.url.replace(/\/+$/,'')}/transcribe`, { method: 'POST', headers: { 'Content-Type': mime||'application/octet-stream' }, body: new Uint8Array(buffer), signal: AbortSignal.timeout(STT_TIMEOUT_MS) });
      if (!res.ok) return null;
      const body = await res.json() as { text?: unknown; language?: unknown; duration?: unknown };
      if (typeof body.text !== 'string') return null;
      return { text: body.text, language: String(body.language??''), duration: Number(body.duration)||0 };
    } catch (e) { this.logger.warn(`stt: ${(e as Error).message}`); return null; }
  }
}
