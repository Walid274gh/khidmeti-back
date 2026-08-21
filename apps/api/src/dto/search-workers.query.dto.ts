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

export class SearchWorkersQueryDto {
  /** Free-text query — French / Arabic / Darija all embed fine. */
  @IsString()
  @Length(2, 300)
  q: string;

  /**
   * Comma-separated wilaya codes ("31,46,38") — the client sends the user's
   * wilaya + its neighbours, the same envelope the map browse uses.
   */
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
