import {
  IsString, IsNotEmpty, IsNumber, IsOptional, IsDate, Min, MaxLength,
} from 'class-validator';
import { Type } from 'class-transformer';

export class CreateBidDto {
  @IsString()
  @IsNotEmpty()
  serviceRequestId: string;

  @IsString()
  @IsNotEmpty()
  workerId: string;

  @IsString()
  @IsNotEmpty()
  workerName: string;

  @IsNumber()
  @Min(0)
  workerAverageRating: number;

  @IsNumber()
  @Min(0)
  workerJobsCompleted: number;

  @IsString()
  @IsOptional()
  workerProfileImageUrl?: string;

  @IsNumber()
  @Min(0)
  proposedPrice: number;

  @IsNumber()
  @Min(1)
  estimatedMinutes: number;

  @IsDate()
  @Type(() => Date)
  availableFrom: Date;

  @IsString()
  @IsOptional()
  @MaxLength(500)
  message?: string;

  @IsDate()
  @IsOptional()
  @Type(() => Date)
  expiresAt?: Date;
}
