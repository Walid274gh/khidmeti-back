import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document } from 'mongoose';

export type NotificationDocument = Notification & Document;

@Schema({ collection: 'notifications', timestamps: false, versionKey: false })
export class Notification {
  @Prop({ required: true })
  _id: string;                         // _id is auto-indexed by MongoDB

  @Prop({ required: true, index: true })
  userId: string;

  @Prop({ required: true })
  title: string;

  @Prop({ required: true })
  body: string;

  @Prop({ required: true, index: true })
  type: string;

  @Prop({ type: Object, default: {} })
  data: Record<string, unknown>;

  @Prop({ required: true, type: Date, index: true })
  createdAt: Date;

  @Prop({ default: false, index: true })
  isRead: boolean;
}

export const NotificationSchema = SchemaFactory.createForClass(Notification);

NotificationSchema.index({ userId: 1, isRead: 1, createdAt: -1 });
