import {
  Body,
  Controller,
  ForbiddenException,
  Get,
  HttpCode,
  HttpStatus,
  Param,
  Post,
  Query,
  UseGuards,
} from '@nestjs/common';
import { IsInt, IsNumber, Max, Min } from 'class-validator';
import { FirebaseAuthGuard }  from '../../common/guards/firebase-auth.guard';
import { CurrentUser }        from '../../common/decorators/current-user.decorator';
import { AuthUser }           from '../../common/guards/firebase-auth.guard';
import { LocationService, AssignCellResult } from './location.service';
import { UserDocument }       from '../../schemas/user.schema';

class AssignCellDto {
  @IsNumber() @Min(-90) @Max(90)   latitude: number;
  @IsNumber() @Min(-180) @Max(180) longitude: number;
  @IsInt() @Min(1) @Max(58)        wilayaCode: number;
}

@Controller('location')
@UseGuards(FirebaseAuthGuard)
export class LocationController {
  constructor(private readonly locationService: LocationService) {}

  /** POST /location/workers/:id/cell */
  @Post('workers/:id/cell')
  @HttpCode(HttpStatus.OK)
  async assignWorkerCell(
    @Param('id') id: string,
    @Body() dto: AssignCellDto,
    @CurrentUser() user: AuthUser,
  ): Promise<AssignCellResult> {
    if (id !== user.uid) throw new ForbiddenException('You can only assign your own cell');
    return this.locationService.assignWorkerToCell(id, dto.latitude, dto.longitude, dto.wilayaCode);
  }

  /** GET /location/cells/:cellId/workers */
  @Get('cells/:cellId/workers')
  async getWorkersInCell(
    @CurrentUser() user: AuthUser,
    @Param('cellId') cellId: string,
    @Query('serviceType') serviceType?: string,
    @Query('onlineOnly')  onlineOnlyStr?: string,
    @Query('limit')       limitStr?: string,
  ): Promise<UserDocument[]> {
    const onlineOnly = onlineOnlyStr === 'true';
    const limit = (limitStr ? parseInt(limitStr, 10) : 50) || 50; // NaN/0 → 50
    return this.locationService.getWorkersInCell(cellId, serviceType, onlineOnly, limit, user.uid);
  }

  /** GET /location/cells/:cellId/adjacent */
  @Get('cells/:cellId/adjacent')
  getAdjacentCells(@Param('cellId') cellId: string): { adjacentCellIds: string[] } {
    return { adjacentCellIds: this.locationService.getAdjacentCellIds(cellId) };
  }
}
