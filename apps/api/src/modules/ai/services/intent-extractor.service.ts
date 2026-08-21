// apps/api/src/modules/ai/services/intent-extractor.service.ts
//
// v15 — 100% auto-hébergé : Gemma4 SUPPRIMÉ.
//
//   texte  → dziriBERT (ai-nlu :8013) + gazetier wilaya
//   audio  → faster-whisper (ai-stt :8014) → chemin texte
//   image  → SigLIP2 zero-shot (ai-vision :8015)
//
// Plus aucun LLM génératif : si un conteneur IA est absent/injoignable, le
// service dégrade en FALLBACK (confidence 0) → l'app bascule sur la recherche
// manuelle (gate 0.35 côté app). Pas de circuit breaker : les services
// NluService/SttService/VisionService retournent déjà null sans jeter.
// Décision utilisateur (2026-08-03) : cible = Codespace 8 GB / 2 cores —
// le conteneur llama.cpp (~5 GB) est infaisable et le fallback violait
// la règle « pas d'API/LLM au runtime ».

import { Inject, Injectable, Logger, Optional } from '@nestjs/common';
import { createHash }                           from 'crypto';
import { AiRateLimitException }                 from '../exceptions/ai-provider.exception';
import { NluService }                           from './nlu.service';
import { SttService }                           from './stt.service';
import { VisionService }                        from './vision.service';
import { findWilaya }                           from '../gazetteer';
import { findProfession }                       from '../profession-gazetteer';
import type { Redis }                           from 'ioredis';

// ── Types publics ──────────────────────────────────────────────────────────────

export interface SearchIntent {
  profession:          string | null;
  is_urgent:           boolean;
  problem_description: string;
  max_radius_km:       number | null;
  confidence:          number;
  transcribedText?:    string;
  // P3 pipeline darija — présents quand dziriBERT/gazetier ont traité le texte
  intent?:             string | null;
  wilaya_code?:        number | null;
  wilaya_name?:        string | null;
}

// ── Constantes ─────────────────────────────────────────────────────────────────

const VALID_PROFESSIONS = new Set([
  'plumber', 'electrician', 'cleaner', 'painter', 'carpenter',
  'ac_repair', 'appliance_repair', 'mason', 'mechanic', 'mover',
  'plasterer', 'welder', 'barber', 'tailor', 'caterer',
]);

const FALLBACK: SearchIntent = {
  profession:          null,
  is_urgent:           false,
  problem_description: '',
  max_radius_km:       null,
  confidence:          0,
};

// ── Détection transcriptions parasites (timestamps Whisper, bruit) ─────────────

const GARBAGE_RE = /^(?:\[\d{1,2}:\d{2}(?:\.\d+)?\s*→?\s*\d{0,2}:?\d{0,2}(?:\.\d+)?\]\s*)+$/;

function isGarbageResponse(text: string): boolean {
  const t = text.trim();
  if (t.length < 3)               return true;
  if (GARBAGE_RE.test(t))         return true;
  if (/^[\d\s:.,\-\[\]→]+$/.test(t)) return true;
  return false;
}

// ══════════════════════════════════════════════════════════════════════════════
// SERVICE PRINCIPAL
// ══════════════════════════════════════════════════════════════════════════════

@Injectable()
export class IntentExtractorService {
  private readonly logger = new Logger(IntentExtractorService.name);

  // ── Cache en mémoire (LRU simple) ─────────────────────────────────────────
  private readonly cache     = new Map<string, SearchIntent>();
  private readonly MAX_CACHE = 200;

  // ── Rate limiting ──────────────────────────────────────────────────────────
  private readonly RATE_LIMIT_MAX    = 20;
  private readonly RATE_LIMIT_WINDOW = 3_600_000; // 1h en ms

  constructor(
    private readonly nlu: NluService,
    private readonly stt: SttService,
    private readonly vision: VisionService,
    @Optional() @Inject('REDIS_CLIENT')
    private readonly redis?: Redis,
  ) {}

  // ── API publique ────────────────────────────────────────────────────────────

  /** Extraction d'intention depuis un texte (Darija / FR / AR / mix) */
  async extractFromText(text: string, uid?: string): Promise<SearchIntent> {
    const trimmed = text.trim().slice(0, 4000);
    if (!trimmed) return { ...FALLBACK };

    if (uid) await this.checkRateLimit(uid);

    const cacheKey = this.hashKey(trimmed.toLowerCase());
    const cached   = this.cache.get(cacheKey);
    if (cached) {
      this.logger.debug(`Cache hit — key=${cacheKey.slice(0, 8)}`);
      return { ...cached };
    }

    // dziriBERT (ai-nlu) — CPU, <1s, comprend le darija.
    // Gazetier wilaya + gazetier MÉTIERS sans modèle. Service absent → FALLBACK
    // (non mis en cache : une panne de conteneur ne doit pas empoisonner le cache).
    const nluRes = await this.nlu.classify(trimmed);
    if (!nluRes) {
      this.logger.warn('ai-nlu indisponible → FALLBACK (recherche manuelle côté app)');
      return { ...FALLBACK };
    }

    const wilaya = findWilaya(trimmed);

    // Rehausseur métiers (bug 20/08) : le NLU ne connaît pas certains mots
    // réels (ex. « بلومبي », absent du lexique appris qui n'a que « بلومبيي »)
    // → confiance < gate 0.35 côté app → 0 % malgré une transcription PARFAITE.
    // Si le gazetier trouve un mot de métier EXPLICITE, il tranche au-dessus du
    // NLU — c'est un nom propre de métier, pas une inférence.
    const gazette = findProfession(trimmed);
    const nluProf =
      VALID_PROFESSIONS.has(nluRes.profession) ? nluRes.profession : null;
    const boostConf =
      gazette && (!nluProf || nluRes.profession_confidence < 0.35) ? 0.85 : null;
    const profession = boostConf ? gazette
                      : nluProf ? nluProf
                      : null;

    const intent: SearchIntent = {
      profession,
      is_urgent:           nluRes.intent === 'urgent_service',
      // Pas de tête générative : le texte brut sert de requête sémantique
      // côté app quand profession=null (nomic-embed est multilingue).
      problem_description: trimmed.slice(0, 120),
      max_radius_km:       null,
      confidence:          boostConf ?? (profession ? nluRes.profession_confidence : 0),
      intent:              nluRes.intent,
      wilaya_code:         wilaya?.code ?? null,
      wilaya_name:         wilaya?.name ?? null,
    };

    this.logger.debug(
      `NLU — intent=${intent.intent} profession=${profession ?? 'null'} ` +
      `conf=${intent.confidence} (nlu=${nluRes.profession_confidence.toFixed(2)} ` +
      `gazetier=${gazette ?? '-'}) wilaya=${wilaya?.code ?? 'null'}`,
    );
    this.setCache(cacheKey, intent);
    return intent;
  }

  /**
   * Extraction depuis un audio : faster-whisper (ai-stt) transcrit puis le
   * chemin texte fait le reste. STT absent ou transcription parasite → FALLBACK.
   */
  async extractFromAudio(buffer: Buffer, mime: string, uid?: string): Promise<SearchIntent> {
    if (uid) await this.checkRateLimit(uid);

    const stt = await this.stt.transcribe(buffer, mime);
    if (!stt) {
      this.logger.warn('ai-stt indisponible → FALLBACK');
      return { ...FALLBACK };
    }

    const text = stt.text.trim();
    if (!text || isGarbageResponse(text)) {
      this.logger.debug('STT: transcription vide ou parasite → FALLBACK');
      return { ...FALLBACK };
    }

    this.logger.debug(
      `STT — lang=${stt.language} ${stt.duration}s → "${text.slice(0, 80)}"`,
    );
    // uid omis : déjà rate-limité ci-dessus.
    const intent = await this.extractFromText(text);
    return { ...intent, transcribedText: text };
  }

  /**
   * Extraction depuis une image : SigLIP2 zero-shot (ai-vision), réponse
   * finale ('none' → profession null → fallback image côté app).
   */
  async extractFromImage(imageBase64: string, uid?: string): Promise<SearchIntent> {
    if (uid) await this.checkRateLimit(uid);

    const vis = await this.vision.classify(Buffer.from(imageBase64, 'base64'));
    if (!vis) {
      this.logger.warn('ai-vision indisponible → FALLBACK');
      return { ...FALLBACK };
    }

    const profession =
      VALID_PROFESSIONS.has(vis.profession) ? vis.profession : null;
    this.logger.debug(
      `Vision — profession=${profession ?? 'null'} conf=${vis.confidence}`,
    );
    return {
      profession,
      is_urgent:           false,
      problem_description: '',
      max_radius_km:       null,
      confidence:          profession ? vis.confidence : 0,
    };
  }

  // ── Cache ──────────────────────────────────────────────────────────────────

  private hashKey(text: string): string {
    return createHash('sha256').update(text).digest('hex').slice(0, 16);
  }

  private setCache(key: string, intent: SearchIntent): void {
    if (this.cache.size >= this.MAX_CACHE) {
      const oldest = this.cache.keys().next().value as string;
      this.cache.delete(oldest);
    }
    this.cache.set(key, intent);
  }

  // ── Rate limiting Redis ────────────────────────────────────────────────────

  private async checkRateLimit(uid: string): Promise<void> {
    if (!this.redis) return;

    const key = `ai_rate:${uid}`;
    const now  = Date.now();

    try {
      const pipeline = this.redis.pipeline();
      pipeline.zremrangebyscore(key, '-inf', now - this.RATE_LIMIT_WINDOW);
      pipeline.zcard(key);
      pipeline.zadd(key, now, `${now}`);
      pipeline.expire(key, 3600);
      const results = await pipeline.exec();
      const count   = (results?.[1]?.[1] as number) ?? 0;

      if (count >= this.RATE_LIMIT_MAX) {
        await this.redis.zrem(key, `${now}`);
        throw new AiRateLimitException();
      }
    } catch (e) {
      if ((e as Error).constructor?.name === 'AiRateLimitException') throw e;
      this.logger.warn(`Redis rate-limit dégradé: ${(e as Error).message}`);
    }
  }
}
