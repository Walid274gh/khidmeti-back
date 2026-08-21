import { IsNumber, IsOptional, IsString, MaxLength, Min } from 'class-validator';

export class CompleteJobDto {
  @IsString()
  @IsOptional()
  @MaxLength(1000)
  workerNotes?: string;

  @IsNumber()
  @IsOptional()
  @Min(0)
  finalPrice?: number;
}
