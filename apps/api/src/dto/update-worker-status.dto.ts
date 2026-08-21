import { IsBoolean } from 'class-validator';

export class UpdateWorkerStatusDto {
  @IsBoolean()
  isOnline: boolean;
}
