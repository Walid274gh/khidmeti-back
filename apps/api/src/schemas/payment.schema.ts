import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document } from 'mongoose';

export type PaymentDocument = Payment & Document;

export type PaymentStatus = 'pending' | 'confirmed' | 'failed' | 'expired';
export type PaymentMethod = 'CIB' | 'EDAHABIA';

@Schema({ collection: 'payments', timestamps: false, versionKey: false })
export class Payment {
  @Prop({ required: true })
  _id!: string;

  @Prop({ required: true, index: true })
  userId!: string;

  @Prop({ required: true })
  tier!: string;

  /** Billing period: 'monthly' (30d) | 'annual' (365d). */
  @Prop({ type: String, default: 'monthly' })
  period!: string;

  /** Days granted on confirm — 30 monthly, 365 annual. */
  @Prop({ type: Number, default: 30 })
  billedDays!: number;

  /** Snapshot of entitlements at intent time — server reprices, never trusts client price */
  @Prop({ type: Object, default: null })
  entitlements!: Record<string, unknown> | null;

  @Prop({ required: true })
  amount!: number; // DZD

  @Prop({ required: true })
  method!: PaymentMethod;

  @Prop({ required: true, unique: true, index: true })
  idempotencyKey!: string;

  @Prop({ required: true, enum: ['pending', 'confirmed', 'failed', 'expired'], default: 'pending', index: true })
  status!: PaymentStatus;

  /** Gateway reference (SATIM order id) */
  @Prop({ type: String, default: null })
  gatewayRef!: string | null;

  /** Last gateway response payload for debugging */
  @Prop({ type: Object, default: null })
  gatewayResponse!: Record<string, unknown> | null;

  @Prop({ required: true, type: Date })
  createdAt!: Date;

  @Prop({ type: Date, default: null })
  confirmedAt!: Date | null;

  @Prop({ type: Date, default: null })
  expiresAt!: Date | null;
}

export const PaymentSchema = SchemaFactory.createForClass(Payment);
PaymentSchema.index({ userId: 1, createdAt: -1 });
PaymentSchema.index({ status: 1, createdAt: 1 });
