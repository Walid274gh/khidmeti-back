import { IsInt, IsOptional, IsString, Max, MaxLength, Min, MinLength } from 'class-validator';

export class SubmitRatingDto {
  @IsInt()
  @Min(1)
  @Max(5)
  stars: number;

  @IsString()
  @IsOptional()
  @MinLength(1)
  @MaxLength(1000)
  comment?: string;
}
