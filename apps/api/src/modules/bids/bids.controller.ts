import {
  Body,
  Controller,
  Get,
  HttpCode,
  HttpStatus,
  Param,
  Post,
  Query,
  UseGuards,
} from '@nestjs/common';
import { FirebaseAuthGuard } from '../../common/guards/firebase-auth.guard';
import { CurrentUser } from '../../common/decorators/current-user.decorator';
import { AuthUser } from '../../common/guards/firebase-auth.guard';
import { BidsService } from './bids.service';
import { CreateBidDto } from '../../dto/create-bid.dto';
import { WorkerBidDocument } from '../../schemas/worker-bid.schema';

@Controller('bids')
@UseGuards(FirebaseAuthGuard)
export class BidsController {
  constructor(private readonly bidsService: BidsService) {}

  /**
   * POST /bids
   * Worker submits a bid on an open service request.
   */
  @Post()
  @HttpCode(HttpStatus.CREATED)
  async submit(
    @Body() dto: CreateBidDto,
    @CurrentUser() user: AuthUser,
  ): Promise<WorkerBidDocument> {
    return this.bidsService.submit(dto, user.uid);
  }

  /**
   * GET /bids
   * List bids. Filtered by serviceRequestId or workerId.
   */
  @Get()
  async findMany(
    @Query('serviceRequestId') serviceRequestId?: string,
    @Query('workerId') workerId?: string,
    @Query('status') status?: string,
    @Query('limit') limitStr?: string,
  ): Promise<WorkerBidDocument[]> {
    const limit = (limitStr ? parseInt(limitStr, 10) : 50) || 50; // NaN/0 → 50
    return this.bidsService.findMany({ serviceRequestId, workerId, status, limit });
  }

  /**
   * GET /bids/:id
   */
  @Get(':id')
  async findById(@Param('id') id: string): Promise<WorkerBidDocument> {
    return this.bidsService.findById(id);
  }

  /**
   * POST /bids/:id/accept
   * Client accepts a bid — transitions request to bidSelected.
   */
  @Post(':id/accept')
  @HttpCode(HttpStatus.NO_CONTENT)
  async accept(
    @Param('id') id: string,
    @CurrentUser() user: AuthUser,
  ): Promise<void> {
    return this.bidsService.accept(id, user.uid);
  }

  /**
   * POST /bids/:id/withdraw
   * Worker withdraws their own pending bid.
   */
  @Post(':id/withdraw')
  @HttpCode(HttpStatus.NO_CONTENT)
  async withdraw(
    @Param('id') id: string,
    @CurrentUser() user: AuthUser,
  ): Promise<void> {
    return this.bidsService.withdraw(id, user.uid);
  }
}
