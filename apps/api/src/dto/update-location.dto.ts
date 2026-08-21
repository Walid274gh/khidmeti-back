import { IsNumber, IsOptional, IsString, Max, Min } from 'class-validator';

export class UpdateLocationDto {
  @IsNumber()
  @Min(-90)
  @Max(90)
  latitude: number;

  @IsNumber()
  @Min(-180)
  @Max(180)
  longitude: number;

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
