import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document } from 'mongoose';

export type ContactInteractionDocument = ContactInteraction & Document;

/**
 * ContactInteraction — the ONE unlock event of the Quote flow.
 *
 * Created exactly once per accepted quote (idempotent on quoteId):
 *   client ACCEPTs quote → contact info unlocked → interaction created.
 *
 * contactedAt = accept time (NOT a call claim — the real call happens
 * outside the app). Rating eligibility derives from these two timestamps:
 *   reminder  = contactedAt + 48h (single nudge, via NotificationsService query)
 *   expiry    = contactedAt + 30d (rateableUntil — silent after that)
 *
 * outcome is write-once (pending → hired | nohire). nohire is a NEUTRAL
 * analytic signal in V1 — never a penalty, never ranked.
 */
@Schema({ collection: 'contact_interactions', timestamps: false, versionKey: false })
export class ContactInteraction {
  @Prop({ required: true })
  _id: string; // _id is auto-indexed by MongoDB

  @Prop({ required: true, index: true })
  clientId: string;

  @Prop({ required: true, index: true })
  workerId: string;

  @Prop({ required: true, default: 'bid_accepted', index: true })
  source: string; // V1: only 'bid_accepted'. Map/directory = visible phone, NO interaction.

  @Prop({ required: true, index: true })
  requestId: string;

  // Denormalized at creation (same pattern as WorkerBid.workerName) — the
  // contacts list renders without a per-row user lookup.
  @Prop({ required: true, default: '' })
  workerName: string;

  @Prop({ required: true, default: '' })
  serviceType: string;

  // Worker avatar at unlock time (same denormalization as workerName) — the
  // contacts list renders without a per-row user lookup. Old rows read null
  // → the UI falls back to monogram initials. Never a phone number: the
  // phone stays with the OS call log, not in this collection.
  @Prop({ type: String, default: null })
  workerProfileImageUrl: string | null;

  @Prop({ required: true, unique: true, index: true })
  quoteId: string; // idempotency key — one interaction per accepted quote, ever.

  @Prop({ required: true, type: Date, index: true })
  contactedAt: Date;

  @Prop({ required: true, type: Date, index: true })
  rateableUntil: Date; // contactedAt + 30d, computed once at creation.

  @Prop({ required: true, default: 'pending', index: true })
  outcome: string; // 'pending' | 'hired' | 'nohire' — write-once.

  @Prop({ default: false, index: true })
  reminderSent: boolean; // 48h nudge sent once — never re-sent.

  @Prop({ type: Date, default: null })
  resolvedAt: Date | null;

  @Prop({ type: Number, default: null, min: 1, max: 5 })
  stars: number | null; // set only when outcome = 'hired'.

  @Prop({ type: [String], default: [] })
  tags: string[]; // set only when outcome = 'hired'. Server-validated whitelist.

  @Prop({ required: true, type: Date, index: true })
  createdAt: Date;
}

export const ContactInteractionSchema = SchemaFactory.createForClass(ContactInteraction);

// Reminder sweep: pending + reminderSent=false + contactedAt <= now-48h + not expired.
// Eligibility check: _id + client ownership + outcome=pending + now <= rateableUntil.
ContactInteractionSchema.index({ outcome: 1, reminderSent: 1, contactedAt: 1 });
ContactInteractionSchema.index({ clientId: 1, createdAt: -1 });
ContactInteractionSchema.index({ workerId: 1, createdAt: -1 });
