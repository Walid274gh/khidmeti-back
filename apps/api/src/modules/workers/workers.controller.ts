import {
  Body,
  Controller,
  ForbiddenException,
  Get,
  HttpCode,
  HttpStatus,
  Param,
  Patch,
  Post,
  Put,
  Query,
  UseGuards,
} from '@nestjs/common';
import { FirebaseAuthGuard, SkipApprovalGate } from '../../common/guards/firebase-auth.guard';
import { CurrentUser }          from '../../common/decorators/current-user.decorator';
import { AuthUser }             from '../../common/guards/firebase-auth.guard';
import { WorkersService }       from './workers.service';
import { UsersService }         from '../users/users.service';
import { UserRole }             from '../../schemas/user.schema';
import { CreateWorkerDto }      from '../../dto/create-worker.dto';
import { UpdateWorkerDto }      from '../../dto/update-worker.dto';
import { UpdateLocationDto }    from '../../dto/update-location.dto';
import { UpdateFcmTokenDto }    from '../../dto/update-fcm-token.dto';
import { UpdateWorkerStatusDto } from '../../dto/update-worker-status.dto';
import { UpdatePortfolioDto }    from '../../dto/update-portfolio.dto';
import { UserDocument }         from '../../schemas/user.schema';

@Controller('workers')
@UseGuards(FirebaseAuthGuard)
export class WorkersController {
  constructor(
    private readonly workersService: WorkersService,
    private readonly usersService: UsersService,
  ) {}

  /** POST /workers — create or update caller's worker profile. */
  // @SkipApprovalGate: a rejected worker resubmits corrected documents here.
  @Post()
  @SkipApprovalGate()
  @HttpCode(HttpStatus.OK)
  async upsert(
    @Body() dto: CreateWorkerDto,
    @CurrentUser() user: AuthUser,
  ): Promise<UserDocument> {
    if (dto.id !== user.uid) throw new ForbiddenException('You can only create your own worker profile');
    return this.workersService.upsert(dto);
  }

  /** GET /workers — list workers (optionally filtered). */
  @Get()
  async findMany(
    @CurrentUser() user: AuthUser,
    @Query('wilayaCode') wilayaCodeStr?: string,
    @Query('profession') profession?: string,
    @Query('isOnline')   isOnlineStr?: string,
    @Query('cellId')     cellId?: string,
    @Query('limit')      limitStr?: string,
  ): Promise<UserDocument[]> {
    const wilayaCode = wilayaCodeStr ? parseInt(wilayaCodeStr, 10) : undefined;
    const isOnline   = isOnlineStr   !== undefined ? isOnlineStr === 'true' : undefined;
    const limit      = limitStr ? Math.min(parseInt(limitStr, 10), 200) : 100;

    // Business-account view is server-enforced from the viewer's persisted role.
    const b2bOnly = (await this.usersService.getRole(user.uid)) === UserRole.Business;

    return this.workersService.findMany({ wilayaCode, profession, isOnline, cellId, limit, b2bOnly });
  }

  @Get(':id')
  async findById(@Param('id') id: string): Promise<UserDocument> {
    return this.workersService.findById(id);
  }

  @Patch(':id')
  async update(
    @Param('id') id: string,
    @Body() dto: UpdateWorkerDto,
    @CurrentUser() user: AuthUser,
  ): Promise<UserDocument> {
    if (id !== user.uid) throw new ForbiddenException('You can only update your own profile');
    return this.workersService.update(id, dto);
  }

  /**
   * PUT /workers/:id/portfolio — replace the showcase gallery with the given
   * ordered list of Cloudinary image URLs (upload the bytes via
   * /media/upload/image first). Returns the updated worker document.
   *
   * 403 SUBSCRIPTION_REQUIRED       — no active pack;
   * 403 PORTFOLIO_QUOTA_EXCEEDED    — more photos than the pack allows.
   */
  @Put(':id/portfolio')
  @HttpCode(HttpStatus.OK)
  async setPortfolio(
    @Param('id') id: string,
    @Body() dto: UpdatePortfolioDto,
    @CurrentUser() user: AuthUser,
  ): Promise<UserDocument> {
    if (id !== user.uid) throw new ForbiddenException('You can only edit your own portfolio');
    return this.workersService.setPortfolio(id, dto.portfolio);
  }

  @Patch(':id/status')
  @HttpCode(HttpStatus.NO_CONTENT)
  async updateStatus(
    @Param('id') id: string,
    @Body() dto: UpdateWorkerStatusDto,
    @CurrentUser() user: AuthUser,
  ): Promise<void> {
    if (id !== user.uid) throw new ForbiddenException('You can only update your own status');
    return this.workersService.updateStatus(id, dto.isOnline);
  }

  @Patch(':id/location')
  @HttpCode(HttpStatus.NO_CONTENT)
  async updateLocation(
    @Param('id') id: string,
    @Body() dto: UpdateLocationDto,
    @CurrentUser() user: AuthUser,
  ): Promise<void> {
    if (id !== user.uid) throw new ForbiddenException('You can only update your own location');
    return this.workersService.updateLocation(
      id, dto.latitude, dto.longitude, dto.cellId, dto.wilayaCode, dto.geoHash,
    );
  }

  @Patch(':id/fcm-token')
  @HttpCode(HttpStatus.NO_CONTENT)
  async updateFcmToken(
    @Param('id') id: string,
    @Body() dto: UpdateFcmTokenDto,
    @CurrentUser() user: AuthUser,
  ): Promise<void> {
    if (id !== user.uid) throw new ForbiddenException('You can only update your own FCM token');
    return this.workersService.updateFcmToken(id, dto.fcmToken);
  }
}
