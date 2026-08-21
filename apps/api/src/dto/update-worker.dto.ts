import { PartialType } from '@nestjs/mapped-types';
import { IsString, IsOptional, IsNumber, IsDate, Min, Max } from 'class-validator';
import { Type } from 'class-transformer';
import { CreateWorkerDto } from './create-worker.dto';

export class UpdateWorkerDto extends PartialType(CreateWorkerDto) {
  @IsString()
  @IsOptional()
  cellId?: string;

  @IsNumber()
  @IsOptional()
  wilayaCode?: number;

  @IsString()
  @IsOptional()
  geoHash?: string;

  @IsOptional()
  @IsDate()
  @Type(() => Date)
  lastCellUpdate?: Date;

  @IsNumber()
  @IsOptional()
  @Min(0)
  @Max(5)
  averageRating?: number;

  @IsNumber()
  @IsOptional()
  @Min(0)
  ratingCount?: number;

  @IsNumber()
  @IsOptional()
  @Min(0)
  ratingSum?: number;

  @IsNumber()
  @IsOptional()
  @Min(0)
  jobsCompleted?: number;

  @IsNumber()
  @IsOptional()
  @Min(0)
  @Max(1)
  responseRate?: number;

  @IsOptional()
  @IsDate()
  @Type(() => Date)
  lastActiveAt?: Date;
}
