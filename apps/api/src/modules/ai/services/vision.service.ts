// apps/api/src/modules/ai/services/vision.service.ts
//
// Client du conteneur `ai-vision` (SigLIP2 zero-shot, tour image ONNX int8 —
// P5 pipeline darija). Image brute (JPEG/PNG/WebP) → { profession, confidence }.
//
// Même contrat de dégradation que NluService/SttService : VISION_URL absent,
// service injoignable ou réponse malformée → null, l'appelant
// (IntentExtractorService) retombe sur FALLBACK (UI image de repli). Aucun
// flux utilisateur ne casse.

import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';

// ViT-B int8 CPU < 2 s par image — 20 s couvre largement démarrage à froid.
const VISION_TIMEOUT_MS = 20_000;

export interface VisionResult {
  profession: string; // classe SigLIP, 'none' inclus
  confidence: number; // sigmoïde SigLIP de la classe gagnante
}

@Injectable()
export class VisionService {
  private readonly logger = new Logger(VisionService.name);
  private readonly url?: string;
  private warnedDisabled = false;

  constructor(config: ConfigService) {
    this.url = config.get<string>('VISION_URL') || undefined;
  }

  get enabled(): boolean {
    return this.url != null;
  }

  /** Classifie [image] — null si le service est désactivé ou injoignable. */
  async classify(image: Buffer): Promise<VisionResult | null> {
    if (!this.url) {
      if (!this.warnedDisabled) {
        this.warnedDisabled = true;
        this.logger.warn('VISION_URL non défini — classification SigLIP désactivée (FALLBACK manuel)');
      }
      return null;
    }

    try {
      const res = await fetch(`${this.url.replace(/\/+$/, '')}/classify`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body:    new Uint8Array(image),
        signal:  AbortSignal.timeout(VISION_TIMEOUT_MS),
      });

      if (!res.ok) {
        this.logger.warn(`ai-vision HTTP ${res.status}`);
        return null;
      }

      const body = (await res.json()) as {
        profession?: unknown;
        profession_confidence?: unknown;
      };
      if (typeof body.profession !== 'string') {
        this.logger.warn('ai-vision réponse malformée');
        return null;
      }

      return {
        profession: body.profession,
        confidence: Number(body.profession_confidence) || 0,
      };
    } catch (err) {
      this.logger.warn(`ai-vision injoignable: ${(err as Error).message}`);
      return null;
    }
  }
}
