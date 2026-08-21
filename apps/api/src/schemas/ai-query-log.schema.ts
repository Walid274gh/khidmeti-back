// ══════════════════════════════════════════════════════════════════════════════
// AiQueryLog — P6 data flywheel (C1, consented logging)
//
// One document per AI extraction (text / voice / photo) from a user who opted
// in (users.aiDataConsent). `chosenProfession` is filled later when the same
// user creates a service request — the human's final choice is the free
// training label (correction signal). Monthly export + retrain = P6.
//
// Raw audio/photo bytes live on disk under FLYWHEEL_DIR (docker volume), the
// document only carries the filename — Mongo never stores media blobs.
// ══════════════════════════════════════════════════════════════════════════════

import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document } from 'mongoose';

export type AiQueryLogDocument = AiQueryLog & Document;

@Schema({ collection: 'ai_query_logs', timestamps: false, versionKey: false })
export class AiQueryLog {
  @Prop({ required: true })
  _id: string; // uuid — also names the media file on disk

  @Prop({ required: true })
  uid: string;

  /** 'text' | 'audio' | 'image' */
  @Prop({ required: true })
  modality: string;

  /** Raw query text, or the STT transcription for audio. '' for images. */
  @Prop({ default: '' })
  text: string;

  // ── Model predictions at query time (accuracy measurement on real data) ────
  @Prop({ type: String, default: null })
  predIntent: string | null;

  @Prop({ type: String, default: null })
  predProfession: string | null;

  @Prop({ default: 0 })
  predConfidence: number;

  @Prop({ type: Number, default: null })
  wilayaCode: number | null;

  /** Filename under FLYWHEEL_DIR ('<_id>.<ext>'). null = no media logged. */
  @Prop({ type: String, default: null })
  mediaFile: string | null;

  // ── Correction label (filled by FlywheelService.recordChoice) ──────────────
  @Prop({ type: String, default: null })
  chosenProfession: string | null;

  @Prop({ type: Date, default: null })
  chosenAt: Date | null;

  @Prop({ required: true })
  createdAt: Date;
}

export const AiQueryLogSchema = SchemaFactory.createForClass(AiQueryLog);

// recordChoice: latest un-labeled log per user within the attribution window.
AiQueryLogSchema.index({ uid: 1, createdAt: -1 });
// Monthly export scans by date range.
AiQueryLogSchema.index({ createdAt: 1 });
