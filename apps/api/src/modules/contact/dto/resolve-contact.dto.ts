import { IsArray, IsIn, IsInt, IsOptional, IsString, ArrayMaxSize, Max, Min } from 'class-validator';

/** Whitelisted quick tags — fixed set, no free text in V1 (anti-extortion). */
export const CONTACT_RATING_TAGS = [
  'on_time',
  'fair_price',
  'clean_work',
  'polite',
  'skilled',
  'would_rehire',
] as const;

export class ResolveContactDto {
  @IsIn(['hired', 'nohire'])
  outcome: string;

  @IsInt()
  @Min(1)
  @Max(5)
  @IsOptional()
  stars?: number;

  @IsArray()
  @IsString({ each: true })
  @IsIn(CONTACT_RATING_TAGS, { each: true })
  @ArrayMaxSize(6)
  @IsOptional()
  tags?: string[];
}
