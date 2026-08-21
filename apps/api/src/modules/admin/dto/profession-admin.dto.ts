// ══════════════════════════════════════════════════════════════════════════════
// Profession CRUD DTOs (admin)
//
// The public ProfessionsController only exposes GET. The admin panel needs full
// create/update, so we validate the trilingual label shape here. Fields mirror
// schemas/profession.schema.ts.
// ══════════════════════════════════════════════════════════════════════════════

import {
  IsBoolean,
  IsEnum,
  IsInt,
  IsNotEmpty,
  IsOptional,
  IsString,
  ValidateNested,
} from 'class-validator';
import { Type } from 'class-transformer';
import { ProfessionCategory } from '../../../schemas/profession.schema';

export class LocalizedLabelDto {
  @IsString() @IsNotEmpty() fr!: string;
  @IsString() @IsNotEmpty() ar!: string;
  @IsString() @IsNotEmpty() en!: string;
}

export class CreateProfessionDto {
  @IsString() @IsNotEmpty() key!: string;

  @IsString() @IsNotEmpty() iconName!: string;

  @IsEnum(ProfessionCategory) categoryKey!: ProfessionCategory;

  @IsOptional() @IsBoolean() isActive?: boolean;

  @IsOptional() @IsInt() sortOrder?: number;

  @ValidateNested() @Type(() => LocalizedLabelDto) labels!: LocalizedLabelDto;

  @ValidateNested() @Type(() => LocalizedLabelDto) categoryLabels!: LocalizedLabelDto;

  @IsOptional() @IsString() imageUrl?: string;
}

export class UpdateProfessionDto {
  @IsOptional() @IsString() iconName?: string;

  @IsOptional() @IsEnum(ProfessionCategory) categoryKey?: ProfessionCategory;

  @IsOptional() @IsBoolean() isActive?: boolean;

  @IsOptional() @IsInt() sortOrder?: number;

  @IsOptional() @ValidateNested() @Type(() => LocalizedLabelDto) labels?: LocalizedLabelDto;

  @IsOptional() @ValidateNested() @Type(() => LocalizedLabelDto) categoryLabels?: LocalizedLabelDto;

  @IsOptional() @IsString() imageUrl?: string;
}
