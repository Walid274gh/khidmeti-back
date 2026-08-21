// apps/api/src/modules/ai/services/stt.service.ts
//
// Client du conteneur `ai-stt` (faster-whisper small int8 — P4 pipeline
// darija). Audio brut (m4a/wav/mp3/ogg — PyAV décode côté conteneur) →
// { text, language, duration }.
//
// Même contrat de dégradation que NluService : STT_URL absent, service
// injoignable ou réponse malformée → null, l'appelant
// (IntentExtractorService) retombe sur FALLBACK (recherche manuelle). Aucun flux
// utilisateur ne casse.

import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';

// whisper-medium int8 CPU ≈ 2-3× temps réel sur les requêtes vocales courtes.
const STT_TIMEOUT_MS = 120_000;
// Au-delà → null → FALLBACK (l'app garde sa borne d'upload existante).
const MAX_STT_BYTES  = 16 * 1024 * 1024;

export interface SttResult {
  text:     string;
  language: string;
  duration: number;
}

@Injectable()
export class SttService {
  private readonly logger = new Logger(SttService.name);
  private readonly url?: string;
  private warnedDisabled = false;

  constructor(config: ConfigService) {
    this.url = config.get<string>('STT_URL') || undefined;
  }

  get enabled(): boolean {
    return this.url != null;
  }

  /** Transcrit [buffer] — null si le service est désactivé ou injoignable. */
  async transcribe(buffer: Buffer, mime: string): Promise<SttResult | null> {
    if (!this.url) {
      if (!this.warnedDisabled) {
        this.warnedDisabled = true;
        this.logger.warn('STT_URL non défini — transcription faster-whisper désactivée (FALLBACK manuel)');
      }
      return null;
    }

    if (buffer.length > MAX_STT_BYTES) {
      this.logger.warn(`Audio ${(buffer.length / 1024 / 1024).toFixed(1)} MB > ${MAX_STT_BYTES / 1024 / 1024} MB — FALLBACK`);
      return null;
    }

    try {
      const res = await fetch(`${this.url.replace(/\/+$/, '')}/transcribe`, {
        method:  'POST',
        headers: { 'Content-Type': mime || 'application/octet-stream' },
        body:    new Uint8Array(buffer),
        signal:  AbortSignal.timeout(STT_TIMEOUT_MS),
      });

      if (!res.ok) {
        this.logger.warn(`ai-stt HTTP ${res.status}`);
        return null;
      }

      const body = (await res.json()) as Partial<SttResult>;
      if (typeof body.text !== 'string') {
        this.logger.warn('ai-stt réponse malformée');
        return null;
      }

      return {
        text:     body.text,
        language: String(body.language ?? ''),
        duration: Number(body.duration) || 0,
      };
    } catch (err) {
      this.logger.warn(`ai-stt injoignable: ${(err as Error).message}`);
      return null;
    }
  }
}
