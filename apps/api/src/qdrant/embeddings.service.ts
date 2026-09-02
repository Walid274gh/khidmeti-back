// apps/api/src/qdrant/embeddings.service.ts
//
// Text → VECTOR_SIZE-dim vectors.
// Priority:
//   1. EMBEDDINGS_URL set → OpenAI-compatible /embeddings (local llama.cpp or Gemini)
//   2. HF_TOKEN set       → HF Inference API (feature-extraction, no model download)
//   3. neither            → disabled (callers skip indexing, graceful degradation)

import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { VECTOR_SIZE } from './qdrant-init.service';

const EMBED_TIMEOUT_MS = 20_000;
const HF_EMBED_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'; // 768-dim

export type EmbedTask = 'document' | 'query';

@Injectable()
export class EmbeddingsService {
  private readonly logger = new Logger(EmbeddingsService.name);
  private readonly url?:    string;
  private readonly model:   string;
  private readonly apiKey?: string;
  private readonly hfToken?: string;
  private warnedDisabled = false;

  constructor(config: ConfigService) {
    this.url     = config.get<string>('EMBEDDINGS_URL') || undefined;
    this.model   = config.get<string>('EMBEDDINGS_MODEL') || 'nomic-embed-text-v1.5';
    this.apiKey  = config.get<string>('EMBEDDINGS_API_KEY') || undefined;
    this.hfToken = config.get<string>('HF_TOKEN') || undefined;
  }

  get enabled(): boolean {
    return this.url != null || this.hfToken != null;
  }

  async embed(text: string, task: EmbedTask = 'document'): Promise<number[] | null> {
    if (!this.url && this.hfToken) {
      return this.embedViaHf(text);
    }
    if (!this.url) {
      if (!this.warnedDisabled) {
        this.warnedDisabled = true;
        this.logger.warn('EMBEDDINGS_URL and HF_TOKEN not set — vector indexing disabled');
      }
      return null;
    }

    const input = this.model.includes('nomic')
      ? `${task === 'document' ? 'search_document' : 'search_query'}: ${text}`
      : text;

    try {
      const res = await fetch(`${this.url.replace(/\/+$/, '')}/embeddings`, {
        method:  'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {}),
        },
        body:   JSON.stringify({ model: this.model, input }),
        signal: AbortSignal.timeout(EMBED_TIMEOUT_MS),
      });

      if (!res.ok) {
        const detail = (await res.text().catch(() => '')).slice(0, 200);
        this.logger.warn(`Embeddings HTTP ${res.status}: ${detail}`);
        return null;
      }

      const body   = (await res.json()) as { data?: Array<{ embedding?: number[] }> };
      const vector = body.data?.[0]?.embedding;

      if (!Array.isArray(vector) || vector.length !== VECTOR_SIZE) {
        this.logger.warn(
          `Embedding dimension mismatch: got ${Array.isArray(vector) ? vector.length : 'none'}, ` +
          `expected ${VECTOR_SIZE} — check EMBEDDINGS_MODEL ("${this.model}")`,
        );
        return null;
      }
      return vector;
    } catch (err) {
      this.logger.warn(`Embeddings call failed: ${(err as Error).message}`);
      return null;
    }
  }

  private async embedViaHf(text: string): Promise<number[] | null> {
    try {
      const res = await fetch(
        `https://api-inference.huggingface.co/pipeline/feature-extraction/${HF_EMBED_MODEL}`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${this.hfToken}`,
          },
          body: JSON.stringify({ inputs: text, options: { wait_for_model: true } }),
          signal: AbortSignal.timeout(EMBED_TIMEOUT_MS),
        },
      );

      if (!res.ok) {
        const detail = (await res.text().catch(() => '')).slice(0, 300);
        if (res.status === 429) {
          this.logger.warn('HF Inference rate limited (429) — embedding skipped');
        } else {
          this.logger.warn(`HF Inference HTTP ${res.status}: ${detail}`);
        }
        return null;
      }

      const vector = (await res.json()) as number[];
      if (!Array.isArray(vector) || vector.length !== VECTOR_SIZE) {
        this.logger.warn(`HF embedding dimension mismatch: got ${Array.isArray(vector) ? vector.length : typeof vector}, expected ${VECTOR_SIZE}`);
        return null;
      }
      return vector;
    } catch (err) {
      this.logger.warn(`HF Inference call failed: ${(err as Error).message}`);
      return null;
    }
  }
}
