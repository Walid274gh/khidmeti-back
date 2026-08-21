import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document } from 'mongoose';
import { BidStatus } from '../common/enums';

export type WorkerBidDocument = WorkerBid & Document;

@Schema({ collection: 'worker_bids', timestamps: false, versionKey: false })
export class WorkerBid {
  @Prop({ required: true })
  _id: string;                         // _id is auto-indexed by MongoDB

  @Prop({ required: true, index: true })
  serviceRequestId: string;

  @Prop({ required: true, index: true })
  workerId: string;

  @Prop({ required: true })
  workerName: string;

  @Prop({ required: true, default: 0.0, min: 0, max: 5 })
  workerAverageRating: number;

  @Prop({ required: true, default: 0, min: 0 })
  workerJobsCompleted: number;

  @Prop({ type: String, default: null })
  workerProfileImageUrl: string | null;

  @Prop({ required: true, min: 0 })
  proposedPrice: number;

  @Prop({ required: true, min: 1 })
  estimatedMinutes: number;

  @Prop({ required: true, type: Date })
  availableFrom: Date;

  @Prop({ type: String, default: null, maxlength: 500 })
  message: string | null;

  @Prop({ required: true, enum: BidStatus, default: BidStatus.Pending, index: true })
  status: BidStatus;

  @Prop({ required: true, type: Date, index: true })
  createdAt: Date;

  @Prop({ type: Date, default: null })
  expiresAt: Date | null;

  @Prop({ type: Date, default: null })
  acceptedAt: Date | null;
}

export const WorkerBidSchema = SchemaFactory.createForClass(WorkerBid);

WorkerBidSchema.index({ serviceRequestId: 1, status: 1 });
WorkerBidSchema.index({ workerId: 1, createdAt: -1 });
// One pending bid per (worker, request). `sparse` is intentionally omitted:
// MongoDB forbids combining it with partialFilterExpression, and the partial
// filter already scopes uniqueness to pending bids only.
WorkerBidSchema.index({ workerId: 1, serviceRequestId: 1 }, {
  unique: true,
  partialFilterExpression: { status: BidStatus.Pending },
});
