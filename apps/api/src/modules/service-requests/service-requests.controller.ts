import {
  Body,
  Controller,
  Get,
  HttpCode,
  HttpStatus,
  Param,
  Patch,
  Post,
  Query,
  UseGuards,
} from '@nestjs/common';
import { FirebaseAuthGuard } from '../../common/guards/firebase-auth.guard';
import { CurrentUser } from '../../common/decorators/current-user.decorator';
import { AuthUser } from '../../common/guards/firebase-auth.guard';
import { ServiceRequestsService } from './service-requests.service';
import { CreateServiceRequestDto } from '../../dto/create-service-request.dto';
import { UpdateServiceRequestDto } from '../../dto/update-service-request.dto';
import { CompleteJobDto } from '../../dto/complete-job.dto';
import { SubmitRatingDto } from '../../dto/submit-rating.dto';
import { ServiceRequestDocument } from '../../schemas/service-request.schema';

@Controller('service-requests')
@UseGuards(FirebaseAuthGuard)
export class ServiceRequestsController {
  constructor(private readonly service: ServiceRequestsService) {}

  /**
   * POST /service-requests
   * Client creates a new service request.
   */
  @Post()
  @HttpCode(HttpStatus.CREATED)
  async create(
    @Body() dto: CreateServiceRequestDto,
    @CurrentUser() user: AuthUser,
  ): Promise<ServiceRequestDocument> {
    return this.service.create(dto, user.uid);
  }

  /**
   * GET /service-requests
   * List requests with filters. Returns up to 50 results.
   * Query: userId, workerId, status, wilayaCode, serviceType, limit
   */
  @Get()
  async findMany(
    @Query('userId') userId?: string,
    @Query('workerId') workerId?: string,
    @Query('status') status?: string,
    @Query('wilayaCode') wilayaCodeStr?: string,
    @Query('serviceType') serviceType?: string,
    @Query('limit') limitStr?: string,
  ): Promise<ServiceRequestDocument[]> {
    // wilayaCode may be comma-separated: "16,09,42" (worker browse expands to
    // neighbouring wilayas). Single value stays a one-element array.
    const wilayaCode = wilayaCodeStr
      ? wilayaCodeStr.split(',').map((s) => parseInt(s, 10)).filter((n) => !isNaN(n))
      : undefined;
    const limit      = (limitStr ? parseInt(limitStr, 10) : 50) || 50; // NaN/0 → 50

    // status may be comma-separated: "open,awaitingSelection"
    const statusFilter: string | string[] | undefined = status
      ? status.includes(',') ? status.split(',') : status
      : undefined;

    return this.service.findMany({ userId, workerId, status: statusFilter, wilayaCode, serviceType, limit });
  }

  /**
   * GET /service-requests/:id
   * Get a single service request.
   */
  @Get(':id')
  async findById(@Param('id') id: string): Promise<ServiceRequestDocument> {
    return this.service.findById(id);
  }

  /**
   * PATCH /service-requests/:id
   * Update a service request (client only, owner check in service layer).
   */
  @Patch(':id')
  async update(
    @Param('id') id: string,
    @Body() dto: UpdateServiceRequestDto,
    @CurrentUser() user: AuthUser,
  ): Promise<ServiceRequestDocument> {
    return this.service.update(id, dto, user.uid);
  }

  /**
   * POST /service-requests/:id/cancel
   * Client cancels their own request.
   */
  @Post(':id/cancel')
  @HttpCode(HttpStatus.NO_CONTENT)
  async cancel(
    @Param('id') id: string,
    @CurrentUser() user: AuthUser,
  ): Promise<void> {
    return this.service.cancel(id, user.uid);
  }

  /**
   * POST /service-requests/:id/decline
   * Assigned worker declines a won job — reopens the request for bids.
   */
  @Post(':id/decline')
  @HttpCode(HttpStatus.NO_CONTENT)
  async decline(
    @Param('id') id: string,
    @CurrentUser() user: AuthUser,
  ): Promise<void> {
    return this.service.decline(id, user.uid);
  }

  /**
   * POST /service-requests/:id/start
   * Assigned worker marks job as in-progress.
   */
  @Post(':id/start')
  @HttpCode(HttpStatus.NO_CONTENT)
  async startJob(
    @Param('id') id: string,
    @CurrentUser() user: AuthUser,
  ): Promise<void> {
    return this.service.startJob(id, user.uid);
  }

  /**
   * POST /service-requests/:id/complete
   * Assigned worker marks job as completed.
   */
  @Post(':id/complete')
  @HttpCode(HttpStatus.NO_CONTENT)
  async completeJob(
    @Param('id') id: string,
    @Body() dto: CompleteJobDto,
    @CurrentUser() user: AuthUser,
  ): Promise<void> {
    return this.service.completeJob(id, user.uid, dto.workerNotes, dto.finalPrice);
  }

  /**
   * POST /service-requests/:id/rate
   * Client submits a star rating after job completion.
   */
  @Post(':id/rate')
  @HttpCode(HttpStatus.NO_CONTENT)
  async submitRating(
    @Param('id') id: string,
    @Body() dto: SubmitRatingDto,
    @CurrentUser() user: AuthUser,
  ): Promise<void> {
    return this.service.submitRating(id, user.uid, dto);
  }
}
