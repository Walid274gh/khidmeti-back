// apps/api/src/qdrant/vector-index.service.ts
//
// The link between MongoDB writes and the Qdrant collections (which stayed at
// 0 points because nothing ever called the upsert helpers). Domain services
// call these hooks fire-and-forget: indexing is best-effort and must NEVER
// fail or slow a user flow — same contract as the WS emits and push sends.
//
// Point IDs — Qdrant only accepts UUIDs or unsigned ints:
//   • service requests already use uuidv4 → used as-is
//   • worker ids are Firebase UIDs → mapped through a deterministic UUIDv5;
//     the raw uid travels in the payload for hydration at search time.
//
// Payloads carry only coarse, non-PII filter fields (no names, no phones —
// only email/phoneNumber are encrypted at rest, but PII stays out of Qdrant
// regardless). Search results must be hydrated from MongoDB.
//
// BACKFILL — data created before this service existed (or while embeddings
// were down) never went through the write hooks. On boot, when a collection
// is empty but MongoDB has matching documents, the gap is filled once in the
// background (sequential — the local embed server is small).

import { Injectable, Logger, OnApplicationBootstrap } from '@nestjs/common';
import { InjectConnection } from '@nestjs/mongoose';
import { Connection } from 'mongoose';
import { v5 as uuidv5 } from 'uuid';
import { EmbeddingsService } from './embeddings.service';
import {
  COLLECTION_REQUESTS,
  COLLECTION_WORKERS,
  QdrantInitService,
} from './qdrant-init.service';

/** Fixed namespace for uid → point-id mapping. NEVER change: ids are stored. */
const WORKER_ID_NAMESPACE = '9a8f6c1e-4b2d-4e7a-b3c5-1d0e8f7a6b5c';

/** Backfill waits for Qdrant collections to exist (init runs in background). */
const BACKFILL_ATTEMPTS = 6;
const BACKFILL_DELAY_MS = 10_000;
/** Safety cap per boot — far above any realistic local/VPS dataset. */
const BACKFILL_MAX_DOCS = 2_000;

interface WorkerLike {
  _id:         string;
  profession?: string | null;
  bio?:        string | null;
  wilayaCode?: number | null;
}

interface RequestLike {
  _id:          string;
  serviceType?: string | null;
  title?:       string | null;
  description?: string | null;
  wilayaCode?:  number | null;
}

@Injectable()
export class VectorIndexService implements OnApplicationBootstrap {
  private readonly logger = new Logger(VectorIndexService.name);

  constructor(
    private readonly embeddings: EmbeddingsService,
    private readonly qdrant:     QdrantInitService,
    @InjectConnection()
    private readonly mongo:      Connection,
  ) {}

  static workerPointId(uid: string): string {
    return uuidv5(uid, WORKER_ID_NAMESPACE);
  }

  onApplicationBootstrap(): void {
    // Background, never blocks boot — mirrors QdrantInitService.onModuleInit.
    void this.backfillIfEmpty();
  }

  // ── Write hooks (called by domain services) ──────────────────────────────

  /** (Re)index a worker profile after create/update. Fire-and-forget. */
  indexWorker(worker: WorkerLike): void {
    void this.tryIndexWorker(worker);
  }

  /** Drop a worker's vector (account deletion). Idempotent, fire-and-forget. */
  removeWorker(uid: string): void {
    void this.qdrant
      .deleteWorkerVector(VectorIndexService.workerPointId(uid))
      .catch((err: Error) => this.logger.warn(`removeWorker(${uid}): ${err.message}`));
  }

  /** (Re)index a service request after create/update. Fire-and-forget. */
  indexRequest(request: RequestLike): void {
    void this.tryIndexRequest(request);
  }

  /** Drop a request's vector (cancelled/completed — never matched again). */
  removeRequest(id: string): void {
    void this.qdrant
      .deleteRequestVector(id)
      .catch((err: Error) => this.logger.warn(`removeRequest(${id}): ${err.message}`));
  }

  // ── Indexing internals ───────────────────────────────────────────────────

  private async tryIndexWorker(worker: WorkerLike): Promise<boolean> {
    try {
      if (!this.embeddings.enabled) return false;
      const text = [worker.profession, worker.bio]
        .filter((s): s is string => !!s && s.trim().length > 0)
        .join('. ');
      if (!text) return false; // nothing semantic to index yet

      const vector = await this.embeddings.embed(text, 'document');
      if (!vector) return false;

      await this.qdrant.upsertWorkerVector(
        VectorIndexService.workerPointId(worker._id),
        vector,
        {
          workerId:   worker._id,
          profession: worker.profession ?? null,
          wilayaCode: worker.wilayaCode ?? null,
        },
      );
      return true;
    } catch (err) {
      this.logger.warn(`indexWorker(${worker._id}) failed: ${(err as Error).message}`);
      return false;
    }
  }

  private async tryIndexRequest(request: RequestLike): Promise<boolean> {
    try {
      if (!this.embeddings.enabled) return false;
      const text = [request.serviceType, request.title, request.description]
        .filter((s): s is string => !!s && s.trim().length > 0)
        .join('. ');
      if (!text) return false;

      const vector = await this.embeddings.embed(text, 'document');
      if (!vector) return false;

      await this.qdrant.upsertRequestVector(request._id, vector, {
        requestId:   request._id,
        serviceType: request.serviceType ?? null,
        wilayaCode:  request.wilayaCode ?? null,
      });
      return true;
    } catch (err) {
      this.logger.warn(`indexRequest(${request._id}) failed: ${(err as Error).message}`);
      return false;
    }
  }

  // ── One-shot backfill ────────────────────────────────────────────────────

  private async backfillIfEmpty(): Promise<void> {
    if (!this.embeddings.enabled) return;

    // Wait for Qdrant + collections (created in background by QdrantInitService).
    let counts: { workers: number; requests: number } | null = null;
    for (let attempt = 1; attempt <= BACKFILL_ATTEMPTS && !counts; attempt++) {
      try {
        const [w, r] = await Promise.all([
          this.qdrant.qdrantClient.count(COLLECTION_WORKERS, { exact: true }),
          this.qdrant.qdrantClient.count(COLLECTION_REQUESTS, { exact: true }),
        ]);
        counts = { workers: w.count, requests: r.count };
      } catch {
        if (attempt === BACKFILL_ATTEMPTS) {
          this.logger.warn('Backfill skipped — Qdrant collections not ready');
          return;
        }
        await new Promise((res) => setTimeout(res, BACKFILL_DELAY_MS));
      }
    }
    if (!counts) return;

    if (counts.workers === 0) {
      const workers = (await this.mongo
        .collection('users')
        .find(
          { role: 'worker', profession: { $nin: [null, ''] } },
          { projection: { _id: 1, profession: 1, bio: 1, wilayaCode: 1 } },
        )
        .limit(BACKFILL_MAX_DOCS)
        .toArray()) as unknown as WorkerLike[];

      let ok = 0;
      for (const w of workers) if (await this.tryIndexWorker(w)) ok++;
      if (workers.length > 0) {
        this.logger.log(`Backfill: indexed ${ok}/${workers.length} worker profiles`);
      }
    }

    if (counts.requests === 0) {
      const requests = (await this.mongo
        .collection('service_requests')
        .find(
          { status: { $nin: ['completed', 'cancelled'] } },
          { projection: { _id: 1, serviceType: 1, title: 1, description: 1, wilayaCode: 1 } },
        )
        .limit(BACKFILL_MAX_DOCS)
        .toArray()) as unknown as RequestLike[];

      let ok = 0;
      for (const r of requests) if (await this.tryIndexRequest(r)) ok++;
      if (requests.length > 0) {
        this.logger.log(`Backfill: indexed ${ok}/${requests.length} open requests`);
      }
    }
  }
}
