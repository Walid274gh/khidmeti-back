// apps/api/src/modules/search/semantic-search.service.ts
//
// Free-text semantic search, both directions: embed the query, k-NN against
// the Qdrant collection, then hydrate through the domain service's canonical
// list query ({ ids }) so results obey the EXACT same visibility contract as
// the structured browse — findWorkers for GET /workers (subscription paywall,
// daily quota, B2B gate), findMany for GET /service-requests (status rules).
// Qdrant supplies candidate ids and relevance ordering only — data always
// comes from MongoDB, so a stale vector can at worst rank a result oddly,
// never resurrect a hidden one.

import {
  Injectable,
  Logger,
  ServiceUnavailableException,
} from '@nestjs/common';
import { EmbeddingsService } from '../../qdrant/embeddings.service';
import { QdrantInitService } from '../../qdrant/qdrant-init.service';
import { UsersService } from '../users/users.service';
import { ServiceRequestsService } from '../service-requests/service-requests.service';
import { ServiceStatus } from '../../common/enums';

/** Cosine floor below which a hit is noise, not a match (nomic-embed scale). */
const MIN_SCORE = 0.35;
/** Hydration filters (paywall / quota / b2b) drop candidates — oversample. */
const OVERSAMPLE = 3;
const MAX_CANDIDATES = 100;

/** A worker document (GET /workers shape) plus its cosine relevance score. */
export type ScoredWorker = Record<string, unknown> & { matchScore: number };

/** A request document (GET /service-requests shape) plus its relevance score. */
export type ScoredRequest = Record<string, unknown> & { matchScore: number };

@Injectable()
export class SemanticSearchService {
  private readonly logger = new Logger(SemanticSearchService.name);

  constructor(
    private readonly embeddings:      EmbeddingsService,
    private readonly qdrant:          QdrantInitService,
    private readonly usersService:    UsersService,
    private readonly requestsService: ServiceRequestsService,
  ) {}

  async searchWorkers(
    query: string,
    opts: { wilayaCodes?: number[]; limit: number; b2bOnly: boolean },
  ): Promise<ScoredWorker[]> {
    const vector = await this.embeddings.embed(query, 'query');
    if (!vector) {
      // Embeddings disabled or down — the app falls back to structured browse.
      throw new ServiceUnavailableException('SEMANTIC_SEARCH_UNAVAILABLE');
    }

    const filter = opts.wilayaCodes?.length
      ? { must: [{ key: 'wilayaCode', match: { any: opts.wilayaCodes } }] }
      : undefined;

    const hits = await this.qdrant.searchWorkers(
      vector,
      filter,
      Math.min(opts.limit * OVERSAMPLE, MAX_CANDIDATES),
      MIN_SCORE,
    );
    if (hits.length === 0) return [];

    // payload.workerId carries the raw uid (point ids are UUIDv5-mapped).
    const scoreByUid = new Map<string, number>();
    for (const h of hits) {
      const uid = h.payload?.['workerId'];
      if (typeof uid === 'string') scoreByUid.set(uid, h.score);
    }
    if (scoreByUid.size === 0) return [];

    // Visibility parity: the same choke point every worker listing goes
    // through — rule changes there apply here automatically.
    const visible = await this.usersService.findWorkers({
      ids:     [...scoreByUid.keys()],
      b2bOnly: opts.b2bOnly,
      limit:   scoreByUid.size,
    });

    return visible
      .map((doc) => ({
        ...(doc.toJSON() as Record<string, unknown>),
        matchScore: scoreByUid.get(doc._id) ?? 0,
      }))
      .sort((a, b) => (b.matchScore as number) - (a.matchScore as number))
      .slice(0, opts.limit) as ScoredWorker[];
  }

  /** Reverse direction: a worker looking for requests that match a skill set. */
  async searchRequests(
    query: string,
    opts: { wilayaCodes?: number[]; limit: number },
  ): Promise<ScoredRequest[]> {
    const vector = await this.embeddings.embed(query, 'query');
    if (!vector) {
      throw new ServiceUnavailableException('SEMANTIC_SEARCH_UNAVAILABLE');
    }

    // `should` (not `must`): geo-untagged requests carry wilayaCode: null in
    // their payload and must stay visible — same rule as the browse, whose
    // Mongo filter is $in: [...codes, null] (see findMany's BUG 3 note).
    const filter = opts.wilayaCodes?.length
      ? {
          should: [
            { key: 'wilayaCode', match: { any: opts.wilayaCodes } },
            { is_null: { key: 'wilayaCode' } },
          ],
        }
      : undefined;

    const hits = await this.qdrant.searchRequests(
      vector,
      filter,
      Math.min(opts.limit * OVERSAMPLE, MAX_CANDIDATES),
      MIN_SCORE,
    );
    if (hits.length === 0) return [];

    const scoreById = new Map<string, number>();
    for (const h of hits) {
      const id = h.payload?.['requestId'];
      if (typeof id === 'string') scoreById.set(id, h.score);
    }
    if (scoreById.size === 0) return [];

    // Visibility parity: hydrate through the same findMany the worker browse
    // uses, with its status envelope — a stale vector for a cancelled or
    // assigned request is filtered out here, never shown.
    const visible = await this.requestsService.findMany({
      ids:    [...scoreById.keys()],
      status: [ServiceStatus.Open, ServiceStatus.AwaitingSelection],
      limit:  scoreById.size,
    });

    return visible
      .map((doc) => ({
        ...(doc.toJSON() as Record<string, unknown>),
        matchScore: scoreById.get(doc._id) ?? 0,
      }))
      .sort((a, b) => (b.matchScore as number) - (a.matchScore as number))
      .slice(0, opts.limit) as ScoredRequest[];
  }
}
