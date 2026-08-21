import { PartialType } from '@nestjs/mapped-types';
import { CreateServiceRequestDto } from './create-service-request.dto';
import {
  IsString, IsOptional, IsNumber, IsDate, IsEnum, Min, Max,
} from 'class-validator';
import { Type } from 'class-transformer';
import { ServiceStatus } from '../common/enums';

export class UpdateServiceRequestDto extends PartialType(CreateServiceRequestDto) {
  @IsEnum(ServiceStatus)
  @IsOptional()
  status?: ServiceStatus;

  @IsString()
  @IsOptional()
  workerId?: string;

  @IsString()
  @IsOptional()
  workerName?: string;

  @IsNumber()
  @IsOptional()
  @Min(0)
  agreedPrice?: number;

  @IsString()
  @IsOptional()
  selectedBidId?: string;

  @IsOptional()
  @IsDate()
  @Type(() => Date)
  bidSelectedAt?: Date;

  @IsOptional()
  @IsDate()
  @Type(() => Date)
  completedAt?: Date;

  @IsString()
  @IsOptional()
  workerNotes?: string;

  @IsNumber()
  @IsOptional()
  @Min(0)
  finalPrice?: number;

  @IsNumber()
  @IsOptional()
  @Min(1)
  @Max(5)
  clientRating?: number;

  @IsString()
  @IsOptional()
  reviewComment?: string;
}
