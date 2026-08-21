// apps/api/src/qdrant/embeddings.service.ts
//
// Text → VECTOR_SIZE-dim vectors over any OpenAI-compatible /embeddings
// endpoint:
//   • local : llama.cpp `ai-embed` container (nomic-embed-text-v1.5,
//             see docker-compose.yml) — EMBEDDINGS_URL=http://ai-embed:8012/v1
//   • cloud : Gemini OpenAI-compat (text-embedding-004, 768-dim) or any other
//             hosted provider — EMBEDDINGS_URL=https://generativelanguage.googleapis.com/v1beta/openai
//
// Config (unset EMBEDDINGS_URL disables vector indexing entirely):
//   EMBEDDINGS_URL      base URL ending in the API root (…/v1)
//   EMBEDDINGS_MODEL    default nomic-embed-text-v1.5
//   EMBEDDINGS_API_KEY  bearer token when the endpoint requires one
//
// Same degradation contract as QdrantInitService: embeddings being down or
// unconfigured must never break a user flow — callers treat null as "skip".

import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { VECTOR_SIZE } from './qdrant-init.service';

const EMBED_TIMEOUT_MS = 20_000;

/** nomic-embed models need asymmetric task prefixes for good retrieval. */
export type EmbedTask = 'document' | 'query';

@Injectable()
export class EmbeddingsService {
  private readonly logger = new Logger(EmbeddingsService.name);
  private readonly url?:    string;
  private readonly model:   string;
  private readonly apiKey?: string;
  private warnedDisabled = false;

  constructor(config: ConfigService) {
    this.url    = config.get<string>('EMBEDDINGS_URL') || undefined;
    this.model  = config.get<string>('EMBEDDINGS_MODEL') || 'nomic-embed-text-v1.5';
    this.apiKey = config.get<string>('EMBEDDINGS_API_KEY') || undefined;
  }

  get enabled(): boolean {
    return this.url != null;
  }

  /**
   * Embeds [text], returning a VECTOR_SIZE-dim vector — or null when the
   * service is disabled, unreachable, or misconfigured (callers skip indexing).
   */
  async embed(text: string, task: EmbedTask = 'document'): Promise<number[] | null> {
    if (!this.url) {
      if (!this.warnedDisabled) {
        this.warnedDisabled = true;
        this.logger.warn('EMBEDDINGS_URL not set — vector indexing disabled');
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
}
