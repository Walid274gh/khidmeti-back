// apps/api/src/modules/ai/services/nlu.service.ts
//
// Client du conteneur `ai-nlu` (dziriBERT 2 têtes ONNX int8 — P3 pipeline
// darija). Texte → { intent, profession, confidences }.
//
// Même contrat de dégradation qu'EmbeddingsService : NLU_URL absent ou
// service injoignable → null, l'appelant (IntentExtractorService) retombe
// en FALLBACK (recherche manuelle côté app). Aucun flux utilisateur ne casse.

import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';

const NLU_TIMEOUT_MS = 8_000;

export interface NluResult {
  intent:                string;
  intent_confidence:     number;
  profession:            string;
  profession_confidence: number;
}

@Injectable()
export class NluService {
  private readonly logger = new Logger(NluService.name);
  private readonly url?: string;
  private warnedDisabled = false;

  constructor(config: ConfigService) {
    this.url = config.get<string>('NLU_URL') || undefined;
  }

  get enabled(): boolean {
    return this.url != null;
  }

  /** Classifie [text] — null si le service est désactivé ou injoignable. */
  async classify(text: string): Promise<NluResult | null> {
    if (!this.url) {
      if (!this.warnedDisabled) {
        this.warnedDisabled = true;
        this.logger.warn('NLU_URL non défini — classification dziriBERT désactivée (FALLBACK manuel)');
      }
      return null;
    }

    try {
      const res = await fetch(`${this.url.replace(/\/+$/, '')}/classify`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ text }),
        signal:  AbortSignal.timeout(NLU_TIMEOUT_MS),
      });

      if (!res.ok) {
        this.logger.warn(`ai-nlu HTTP ${res.status}`);
        return null;
      }

      const body = (await res.json()) as Partial<NluResult>;
      if (typeof body.intent !== 'string' || typeof body.profession !== 'string') {
        this.logger.warn('ai-nlu réponse malformée');
        return null;
      }

      return {
        intent:                body.intent,
        intent_confidence:     Number(body.intent_confidence)     || 0,
        profession:            body.profession,
        profession_confidence: Number(body.profession_confidence) || 0,
      };
    } catch (err) {
      this.logger.warn(`ai-nlu injoignable: ${(err as Error).message}`);
      return null;
    }
  }
}
