import {
  IsString, IsNotEmpty, IsNumber, IsArray, IsOptional,
  IsEnum, IsDate, Min, Max, MinLength, MaxLength,
} from 'class-validator';
import { Type } from 'class-transformer';
import { ServicePriority } from '../common/enums';

export class CreateServiceRequestDto {
  @IsString()
  @IsNotEmpty()
  userId: string;

  @IsString()
  @IsNotEmpty()
  userName: string;

  @IsString()
  @IsNotEmpty()
  userPhone: string;

  @IsString()
  @IsNotEmpty()
  serviceType: string;

  @IsString()
  @IsNotEmpty()
  @MinLength(3)
  @MaxLength(100)
  title: string;

  @IsString()
  @IsNotEmpty()
  @MinLength(10)
  @MaxLength(1000)
  description: string;

  @IsDate()
  @Type(() => Date)
  scheduledDate: Date;

  @IsNumber()
  @Min(0)
  @Max(23)
  scheduledHour: number;

  @IsNumber()
  @Min(0)
  @Max(59)
  scheduledMinute: number;

  @IsEnum(ServicePriority)
  @IsOptional()
  priority?: ServicePriority;

  @IsNumber()
  @Min(-90)
  @Max(90)
  userLatitude: number;

  @IsNumber()
  @Min(-180)
  @Max(180)
  userLongitude: number;

  @IsString()
  @IsNotEmpty()
  userAddress: string;

  @IsArray()
  @IsString({ each: true })
  @IsOptional()
  mediaUrls?: string[];

  @IsNumber()
  @IsOptional()
  @Min(0)
  budgetMin?: number;

  @IsNumber()
  @IsOptional()
  @Min(0)
  budgetMax?: number;

  @IsString()
  @IsOptional()
  cellId?: string;

  @IsNumber()
  @IsOptional()
  wilayaCode?: number;

  @IsString()
  @IsOptional()
  geoHash?: string;
}
