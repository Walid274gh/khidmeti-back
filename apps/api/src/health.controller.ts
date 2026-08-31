import {
  Controller,
  Get,
  HttpCode,
  HttpStatus,
  Inject,
  Optional,
  ServiceUnavailableException,
} from '@nestjs/common';
import { InjectConnection } from '@nestjs/mongoose';
import { Connection } from 'mongoose';
import type { Redis } from 'ioredis';
import { QdrantInitService } from './qdrant/qdrant-init.service';
import { ConfigService } from '@nestjs/config';

type DepState = 'up' | 'down' | 'disabled';

interface ReadyReport {
  status: 'ok' | 'unavailable';
  dependencies: {
    mongodb: DepState; // critique  → 503 si down
    qdrant: DepState; //  dégradable (recherche vectorielle)
    redis: DepState; //   optionnel (rate-limiting)
  };
  timestamp: string;
}

@Controller('health')
export class HealthController {
  constructor(
    @InjectConnection() private readonly mongo: Connection,
    private readonly qdrant: QdrantInitService,
    @Optional() @Inject('REDIS_CLIENT') private readonly redis: Redis | null,
    private readonly config: ConfigService,
  ) {
    this.initHfKeepalive();
  }

  // ── Liveness ────────────────────────────────────────────────────────────────
  // Le process NestJS répond. Utilisé par le healthcheck Docker et nginx.
  // Ne dépend d'AUCUN service externe → reste 200 même si le cloud est down,
  // pour éviter que l'orchestrateur ne tue un conteneur pourtant vivant.
  @Get()
  @HttpCode(HttpStatus.OK)
  check(): { status: string; timestamp: string } {
    return { status: 'ok', timestamp: new Date().toISOString() };
  }

  // ── Readiness ───────────────────────────────────────────────────────────────
  // « Cette instance peut-elle servir le trafic ? »
  //   • MongoDB down  → 503 (dépendance critique)
  //   • Qdrant/Redis  → signalés mais NE bloquent PAS (dégradation gracieuse)
  // Les trois sondes tournent en parallèle et sont bornées dans le temps.
  @Get('ready')
  async ready(): Promise<ReadyReport> {
    const [mongodb, qdrant, redis] = await Promise.all([
      this.checkMongo(),
      this.checkQdrant(),
      this.checkRedis(),
    ]);

    const critical = mongodb === 'up';
    const report: ReadyReport = {
      status: critical ? 'ok' : 'unavailable',
      dependencies: { mongodb, qdrant, redis },
      timestamp: new Date().toISOString(),
    };

    if (!critical) throw new ServiceUnavailableException(report);
    return report;
  }

  // ── Sondes (isolées, sans jeter, bornées dans le temps) ──────────────────────

  private async checkMongo(): Promise<DepState> {
    try {
      if (this.mongo.readyState !== 1 || !this.mongo.db) return 'down';
      await this.withTimeout(this.mongo.db.admin().ping(), 3000);
      return 'up';
    } catch {
      return 'down';
    }
  }

  private async checkQdrant(): Promise<DepState> {
    try {
      return (await this.withTimeout(this.qdrant.isReachable(), 3000)) ? 'up' : 'down';
    } catch {
      return 'down';
    }
  }

  private async checkRedis(): Promise<DepState> {
    if (!this.redis) return 'disabled';
    try {
      const pong = await this.withTimeout(this.redis.ping(), 3000);
      return pong === 'PONG' ? 'up' : 'down';
    } catch {
      return 'down';
    }
  }

  // ── HF keepalive + status (UptimeRobot watches /health + /hf-status) ─────────
  private readonly hfUrl?: string;
  private hfLast: { state: DepState; at: string; ms?: number } = { state: 'down', at: new Date().toISOString() };

  private initHfKeepalive(): void {
    this.hfUrl = this.config.get<string>('HF_SPACE_URL')?.replace(/\/+$/, '');
    if (!this.hfUrl) return;
    // Probe once at boot, then every 50 min (ZeroGPU sleeps after 48 min without queue hit)
    const probe = async () => {
      const t0 = Date.now();
      try {
        const res = await fetch(this.hfUrl + '/config', { signal: AbortSignal.timeout(8000) } as RequestInit);
        this.hfLast = { state: res.ok ? 'up' : 'down', at: new Date().toISOString(), ms: Date.now() - t0 };
      } catch {
        this.hfLast = { state: 'down', at: new Date().toISOString(), ms: Date.now() - t0 };
      }
    };
    probe();
    setInterval(probe, 50 * 60 * 1000);
  }

  @Get('hf-status')
  @HttpCode(HttpStatus.OK)
  async hfStatus(): Promise<{ hf: DepState; lastCheck: string; latencyMs?: number; url?: string }> {
    // On-demand probe so UptimeRobot sees fresh state even between keepalive ticks
    if (this.hfUrl) {
      const t0 = Date.now();
      try {
        const res = await fetch(this.hfUrl + '/config', { signal: AbortSignal.timeout(6000) } as RequestInit);
        this.hfLast = { state: res.ok ? 'up' : 'down', at: new Date().toISOString(), ms: Date.now() - t0 };
      } catch {
        this.hfLast = { state: 'down', at: new Date().toISOString(), ms: Date.now() - t0 };
      }
    }
    return { hf: this.hfLast.state, lastCheck: this.hfLast.at, latencyMs: this.hfLast.ms, url: this.hfUrl };
  }

  /**
   * Course entre la sonde et un délai. Le gagnant règle le résultat ;
   * le perdant est TOUJOURS consommé (then/onRejected attaché) → jamais
   * d'unhandledRejection même si la dépendance répond après le timeout.
   */
  private withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('health check timeout')), ms);
      p.then(
        (value) => {
          clearTimeout(timer);
          resolve(value);
        },
        (err) => {
          clearTimeout(timer);
          reject(err);
        },
      );
    });
  }
}
