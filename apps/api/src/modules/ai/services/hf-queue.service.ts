// hf-queue.service — Gradio queue HTTP client for HF ZeroGPU Space
// The Space runs `sdk: gradio` with @spaces.GPU handlers via Blocks —
// REST /health etc not available. We call it through /queue/join + /queue/data SSE.

import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';

const QUEUE_TIMEOUT_MS = 120_000;

@Injectable()
export class HfQueueService {
  private readonly logger = new Logger(HfQueueService.name);
  private readonly base?: string;
  private warnedNoBase = false;

  constructor(config: ConfigService) {
    this.base = config.get<string>('HF_SPACE_URL')?.replace(/\/+$/, '') || undefined;
  }

  get enabled(): boolean { return !!this.base; }

  // Gradio queue: join (fn_index 0=STT, 1=Vision, 2=NLU) then poll SSE
  async call(fnIndex: number, data: unknown[]): Promise<string> {
    if (!this.base) {
      if (!this.warnedNoBase) { this.warnedNoBase = true; this.logger.warn('HF_SPACE_URL not set'); }
      throw new Error('HF queue disabled');
    }
    const session = `kh-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    const joinRes = await fetch(`${this.base}/queue/join`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data, fn_index: fnIndex, session_hash: session }),
      signal: AbortSignal.timeout(15_000),
    });
    if (!joinRes.ok) throw new Error(`queue join ${joinRes.status}`);
    const { event_id } = await joinRes.json() as { event_id: string };

    const deadline = Date.now() + QUEUE_TIMEOUT_MS;
    while (Date.now() < deadline) {
      const sseRes = await fetch(`${this.base}/queue/data?session_hash=${session}`, {
        headers: { Accept: 'text/event-stream' },
        signal: AbortSignal.timeout(15_000),
      });
      const text = await sseRes.text();
      for (const line of text.split('\n')) {
        if (!line.startsWith('data: ')) continue;
        const msg = JSON.parse(line.slice(6)) as { msg: string; event_id?: string; output?: { data: unknown[] }; success?: boolean };
        if (msg.msg === 'process_completed' && msg.event_id === event_id) {
          if (!msg.success) throw new Error('queue process failed');
          return String(msg.output?.data?.[0] ?? '');
        }
        if (msg.msg === 'unexpected_error') throw new Error(String((msg as Record<string, unknown>).message ?? 'queue error'));
      }
      await new Promise(r => setTimeout(r, 800));
    }
    throw new Error('queue timeout');
  }
}
