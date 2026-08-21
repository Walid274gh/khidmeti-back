import { Transform } from 'class-transformer';
import {
  ArrayMaxSize,
  IsInt,
  IsOptional,
  IsString,
  Length,
  Max,
  Min,
} from 'class-validator';

export class SearchRequestsQueryDto {
  /**
   * Free-text query. OPTIONAL — when absent the worker's own profile
   * (profession + bio) becomes the query vector, turning the endpoint into
   * "jobs that match me" recommendations with no typing required.
   */
  @IsOptional()
  @IsString()
  @Length(2, 300)
  q?: string;

  /** Comma-separated wilaya codes — worker's own wilaya + neighbours. */
  @IsOptional()
  @Transform(({ value }) =>
    typeof value === 'string' && value.trim() !== ''
      ? value.split(',').map((s) => parseInt(s.trim(), 10))
      : undefined)
  @IsInt({ each: true })
  @ArrayMaxSize(10)
  wilayaCodes?: number[];

  @IsOptional()
  @IsInt()
  @Min(1)
  @Max(50)
  limit?: number;
}
