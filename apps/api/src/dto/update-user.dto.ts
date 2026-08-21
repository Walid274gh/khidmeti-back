import { PartialType } from '@nestjs/mapped-types';
import { IsString, IsOptional, IsNumber } from 'class-validator';
import { CreateUserDto } from './create-user.dto';

export class UpdateUserDto extends PartialType(CreateUserDto) {
  @IsString()
  @IsOptional()
  cellId?: string;

  @IsString()
  @IsOptional()
  geoHash?: string;

  @IsNumber()
  @IsOptional()
  wilayaCode?: number;
}
