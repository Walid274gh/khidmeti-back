import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { HfQueueService } from './hf-queue.service';
const NLU_TIMEOUT_MS = 8_000;
export interface NluResult { intent: string; intent_confidence: number; profession: string; profession_confidence: number; }
@Injectable()
export class NluService {
  private readonly logger = new Logger(NluService.name);
  private readonly url?: string;
  constructor(private readonly config: ConfigService, private readonly hf: HfQueueService) { this.url = this.config.get<string>('NLU_URL') || undefined; }
  async classify(text: string): Promise<NluResult | null> {
    if (this.hf.enabled) {
      try {
        const raw = await this.hf.call(2, [text]);
        const body = JSON.parse(raw) as Partial<NluResult>;
        if (typeof body.intent === 'string' && typeof body.profession === 'string')
          return { intent: body.intent, intent_confidence: Number(body.intent_confidence)||0, profession: body.profession, profession_confidence: Number(body.profession_confidence)||0 };
      } catch (e) { this.logger.warn(`HF nlu: ${(e as Error).message}`); }
    }
    if (!this.url) return null;
    try {
      const res = await fetch(`${this.url.replace(/\/+$/,'')}/classify`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }), signal: AbortSignal.timeout(NLU_TIMEOUT_MS) });
      if (!res.ok) return null;
      const body = await res.json() as Partial<NluResult>;
      if (typeof body.intent !== 'string' || typeof body.profession !== 'string') return null;
      return { intent: body.intent, intent_confidence: Number(body.intent_confidence)||0, profession: body.profession, profession_confidence: Number(body.profession_confidence)||0 };
    } catch (e) { this.logger.warn(`nlu: ${(e as Error).message}`); return null; }
  }
}
