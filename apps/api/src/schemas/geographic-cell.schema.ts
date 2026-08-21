import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document } from 'mongoose';

export type GeographicCellDocument = GeographicCell & Document;

@Schema({ collection: 'geographic_cells', timestamps: false, versionKey: false })
export class GeographicCell {
  @Prop({ required: true })
  _id: string;                         // _id is auto-indexed by MongoDB

  @Prop({ required: true })
  wilayaCode: number;

  @Prop({ required: true })
  centerLat: number;

  @Prop({ required: true })
  centerLng: number;

  @Prop({ required: true, default: 5.0 })
  radius: number;

  @Prop({ type: [String], default: [] })
  adjacentCellIds: string[];
}

export const GeographicCellSchema = SchemaFactory.createForClass(GeographicCell);

GeographicCellSchema.index({ wilayaCode: 1 });
