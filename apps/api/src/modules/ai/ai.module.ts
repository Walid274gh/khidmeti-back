// apps/api/src/modules/ai/ai.module.ts
//
// v15 — 100% auto-hébergé : Gemma4 SUPPRIMÉ (décision 2026-08-03).
//   texte = ai-nlu (dziriBERT), audio = ai-stt (faster-whisper),
//   image = ai-vision (SigLIP2). Chaque service dégrade en null → FALLBACK.

import { Module }  from '@nestjs/common';
import { IntentExtractorService } from './services/intent-extractor.service';
import { NluService }    from './services/nlu.service';
import { SttService }    from './services/stt.service';
import { VisionService } from './services/vision.service';
import { FlywheelService } from './services/flywheel.service';
import { HfQueueService } from './services/hf-queue.service';
import { AiController } from './ai.controller';
import { AuthModule }   from '../auth/auth.module';
import Redis            from 'ioredis';

@Module({
  imports:     [AuthModule],
  controllers: [AiController],
  providers: [
    // ── Redis — rate-limiting (dégradation gracieuse si absent) ───────────────
    {
      provide:    'REDIS_CLIENT',
      useFactory: (): Redis | null => {
        const url = process.env['REDIS_URL'];
        if (!url) return null;
        const client = new Redis(url, {
          lazyConnect:          true,
          maxRetriesPerRequest: 1,
          enableOfflineQueue:   false,
        });
        client.on('error', () => { /* silent — Redis est optionnel */ });
        return client;
      },
    },

    IntentExtractorService,
    // P3 pipeline darija — dziriBERT 2 têtes via conteneur ai-nlu
    NluService,
    // P4 pipeline darija — faster-whisper small int8 via conteneur ai-stt
    SttService,
    // P5 pipeline darija — SigLIP2 zero-shot via conteneur ai-vision
    VisionService,
    // C1 flywheel — consented training-data logging (P6)
    FlywheelService,
    // HF ZeroGPU queue client (Gradio Blocks queue — best practice for sdk: gradio)
    HfQueueService,
  ],
  exports: [IntentExtractorService, 'REDIS_CLIENT', FlywheelService, HfQueueService],
})
export class AiModule {}
