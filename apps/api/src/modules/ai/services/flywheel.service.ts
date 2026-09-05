// apps/api/src/modules/ai/services/flywheel.service.ts
//
// P6 data flywheel — C1: consented logging of AI queries as training data.
//
// Two write paths, both fire-and-forget (a logging failure must NEVER break
// search or request creation — same principle as safeEmitUpdated):
//
//   log()          — called by AiController after every extraction. Gated on
//                    users.aiDataConsent (opt-in). Media bytes (voice/photo)
//                    go to FLYWHEEL_DIR on disk; Mongo gets text + predictions.
//   recordChoice() — called by ServiceRequestsService.create. The profession
//                    the user finally committed to labels their latest
//                    un-labeled AI query within the attribution window —
//                    human corrections are the free training labels.

import { Injectable, Logger } from '@nestjs/common';
import { InjectModel }        from '@nestjs/mongoose';
import { Model }              from 'mongoose';
import { randomUUID }         from 'crypto';
import { promises as fs }     from 'fs';
import { join }               from 'path';
import { AiQueryLog, AiQueryLogDocument } from '../../../schemas/ai-query-log.schema';
import { User, UserDocument }             from '../../../schemas/user.schema';
import type { SearchIntent }              from './intent-extractor.service';

/** A service request created this long after an AI query labels it. */
const CHOICE_WINDOW_MS = 30 * 60 * 1000;

/** Media larger than this is skipped (text row still logged) — bounds disk. */
const MAX_MEDIA_BYTES = 16 * 1024 * 1024;

/** Capture telemetry from the Flutter VAD (audio rows only, all optional). */
export interface VoiceMeta {
  vadSpeechSecs?: number;
  vadPeakDb?: number;
  interimText?: string;
  audioBytes?: number;
}

@Injectable()
export class FlywheelService {
  private readonly logger = new Logger(FlywheelService.name);
  private readonly dir    = process.env['FLYWHEEL_DIR'] ?? 'flywheel';

  // ponytail: 60s consent cache — one user read per uid per minute instead of
  // per query. Revoking consent can lag logging by up to 60s; acceptable.
  private readonly consentCache = new Map<string, { v: boolean; exp: number }>();

  constructor(
    @InjectModel(AiQueryLog.name)
    private readonly logModel: Model<AiQueryLogDocument>,
    @InjectModel(User.name)
    private readonly userModel: Model<UserDocument>,
  ) {}

  /** Fire-and-forget: consent check + insert + media write. Never throws. */
  log(
    uid:      string,
    modality: 'text' | 'audio' | 'image',
    text:     string,
    intent:   SearchIntent,
    media?:   Buffer,
    ext?:     string,
    voiceMeta?: VoiceMeta,
  ): void {
    void this.write(uid, modality, text, intent, media, ext, voiceMeta).catch((e) =>
      this.logger.warn(`flywheel log failed (non-fatal): ${(e as Error).message}`),
    );
  }

  /** Fire-and-forget: label the latest un-labeled recent query. Never throws. */
  recordChoice(uid: string, profession: string): void {
    if (!profession) return;
    void this.logModel
      .findOneAndUpdate(
        {
          uid,
          chosenProfession: null,
          createdAt: { $gt: new Date(Date.now() - CHOICE_WINDOW_MS) },
        },
        { chosenProfession: profession, chosenAt: new Date() },
        { sort: { createdAt: -1 } },
      )
      .exec()
      .catch((e) =>
        this.logger.warn(`flywheel choice failed (non-fatal): ${(e as Error).message}`),
      );
  }

  // ── Internals ──────────────────────────────────────────────────────────────

  private async hasConsent(uid: string): Promise<boolean> {
    const cached = this.consentCache.get(uid);
    if (cached && cached.exp > Date.now()) return cached.v;

    const doc = await this.userModel
      .findById(uid)
      .select('aiDataConsent')
      .lean()
      .exec();
    const v = doc?.aiDataConsent === true;
    this.consentCache.set(uid, { v, exp: Date.now() + 60_000 });
    return v;
  }

  private async write(
    uid:      string,
    modality: string,
    text:     string,
    intent:   SearchIntent,
    media?:   Buffer,
    ext?:     string,
    voiceMeta?: VoiceMeta,
  ): Promise<void> {
    if (!(await this.hasConsent(uid))) return;

    const id = randomUUID();
    let mediaFile: string | null = null;

    if (media && media.length > 0 && media.length <= MAX_MEDIA_BYTES) {
      const safeExt = (ext ?? 'bin').replace(/[^a-z0-9]/gi, '').slice(0, 8) || 'bin';
      mediaFile = `${id}.${safeExt}`;
      await fs.mkdir(this.dir, { recursive: true });
      await fs.writeFile(join(this.dir, mediaFile), media);
    }

    await this.logModel.create({
      _id: id,
      uid,
      modality,
      text: (text ?? '').slice(0, 4000),
      predIntent:       intent.intent ?? null,
      predProfession:   intent.profession,
      predConfidence:   intent.confidence,
      wilayaCode:       intent.wilaya_code ?? null,
      mediaFile,
      chosenProfession: null,
      chosenAt:         null,
      vadSpeechSecs:    voiceMeta?.vadSpeechSecs ?? null,
      vadPeakDb:        voiceMeta?.vadPeakDb ?? null,
      interimText:      (voiceMeta?.interimText ?? '').slice(0, 500) || null,
      audioBytes:       voiceMeta?.audioBytes ?? null,
      createdAt:        new Date(),
    });
  }
}
