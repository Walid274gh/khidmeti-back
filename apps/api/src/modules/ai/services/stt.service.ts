import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
const STT_TIMEOUT_MS = 120_000;
export interface SttResult { text: string; language: string; duration: number; }
@Injectable()
export class SttService {
  private readonly logger = new Logger(SttService.name);
  private readonly hfUrl?: string;
  private readonly url?: string;
  constructor(private readonly config: ConfigService) {
    this.hfUrl = this.config.get<string>('HF_SPACE_URL')?.replace(/\/+$/, '');
    this.url = this.config.get<string>('STT_URL') || undefined;
  }
  get enabled(): boolean { return !!(this.hfUrl || this.url); }
  async transcribe(buffer: Buffer, mime: string): Promise<SttResult | null> {
    if (this.hfUrl) {
      try {
        const blob = new Blob([new Uint8Array(buffer)], { type: mime || 'audio/webm' });
        const fd = new FormData(); fd.append('file', blob, 'audio.webm');
        const res = await fetch(`${this.hfUrl}/stt`, { method: 'POST', body: fd as unknown as BodyInit, signal: AbortSignal.timeout(STT_TIMEOUT_MS) });
        if (res.ok) {
          const body = await res.json() as Partial<SttResult>;
          if (typeof body.text === 'string') return { text: body.text, language: String(body.language??'ar'), duration: Number(body.duration)||0 };
        }
      } catch (e) { this.logger.warn(`HF stt: ${(e as Error).message}`); }
    }
    if (!this.url) return null;
    try {
      const res = await fetch(`${this.url.replace(/\/+$/,'')}/transcribe`, { method: 'POST', headers: { 'Content-Type': mime||'application/octet-stream' }, body: new Uint8Array(buffer), signal: AbortSignal.timeout(STT_TIMEOUT_MS) });
      if (!res.ok) return null;
      const body = await res.json() as Partial<SttResult>;
      if (typeof body.text !== 'string') return null;
      return { text: body.text, language: String(body.language??''), duration: Number(body.duration)||0 };
    } catch (e) { this.logger.warn(`stt: ${(e as Error).message}`); return null; }
  }
}
