import {
  BadRequestException,
  ForbiddenException,
  Injectable,
  Logger,
  NotFoundException,
} from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { v4 as uuidv4 } from 'uuid';
import { WorkerBid, WorkerBidDocument }         from '../../schemas/worker-bid.schema';
import { ServiceRequest, ServiceRequestDocument } from '../../schemas/service-request.schema';
import { CreateBidDto }    from '../../dto/create-bid.dto';
import { BidStatus, ServiceStatus } from '../../common/enums';
import { BidsGateway }           from '../gateway/bids.gateway';
import { ServiceRequestGateway } from '../gateway/service-request.gateway';
import { PushSenderService }     from '../notifications/push-sender.service';
import { UsersService }          from '../users/users.service';

export interface BidFilters {
  serviceRequestId?: string;
  workerId?: string;
  status?: string;
  limit?: number;
}

// Bid TTL: accepted bids expire 7 days after acceptance.
// Worker must start the job within this window or contact client for a new bid.
const BID_TTL_DAYS = 7;

@Injectable()
export class BidsService {
  private readonly logger = new Logger(BidsService.name);

  constructor(
    @InjectModel(WorkerBid.name)
    private readonly bidModel: Model<WorkerBidDocument>,
    @InjectModel(ServiceRequest.name)
    private readonly requestModel: Model<ServiceRequestDocument>,
    private readonly bidsGateway: BidsGateway,
    private readonly requestGateway: ServiceRequestGateway,
    private readonly pushSender: PushSenderService,
    private readonly usersService: UsersService,
  ) {}

  /**
   * Fire-and-forget realtime notify. Never let a WebSocket error break the
   * REST mutation — the DB write is already the source of truth.
   */
  private safeEmit(fn: () => void): void {
    try {
      fn();
    } catch (err) {
      this.logger.warn(`Realtime emit failed (non-fatal): ${(err as Error).message}`);
    }
  }

  async submit(dto: CreateBidDto, uid: string): Promise<WorkerBidDocument> {
    let bidConsumed = false;
    try {
      if (dto.workerId !== uid) throw new ForbiddenException('workerId must match authenticated user');

      // Bid gate: atomically consume one bid from the worker's monthly quota.
      // Also enforces the active-subscription rule (SUBSCRIPTION_REQUIRED /
      // BID_NOT_INCLUDED / BID_QUOTA_EXHAUSTED — the app maps each to a CTA).
      // Consumed BEFORE the request checks; refunded on any later failure.
      await this.usersService.consumeBid(uid);
      bidConsumed = true;

      const worker = await this.usersService.findByIdOrNull(uid);
      if (!worker) throw new ForbiddenException('SUBSCRIPTION_REQUIRED');

      const request = await this.requestModel.findById(dto.serviceRequestId).exec();
      if (!request) throw new NotFoundException(`Service request ${dto.serviceRequestId} not found`);
      if (
        request.status !== ServiceStatus.Open &&
        request.status !== ServiceStatus.AwaitingSelection
      ) {
        throw new BadRequestException(`Request is not accepting bids (status: ${request.status})`);
      }

      if (request.userId === uid) throw new ForbiddenException('You cannot bid on your own service request');

      const existingBid = await this.bidModel
        .findOne({ serviceRequestId: dto.serviceRequestId, workerId: uid, status: BidStatus.Pending })
        .exec();
      if (existingBid) throw new BadRequestException('You already have a pending bid on this request');

      const bid = new this.bidModel({
        _id: uuidv4(),
        ...dto,
        // Denormalized worker card stats come from the DB, never the client —
        // these numbers drive the hiring decision.
        workerName:            worker.name,
        workerAverageRating:   worker.averageRating ?? 0,
        workerJobsCompleted:   worker.jobsCompleted ?? 0,
        workerProfileImageUrl: worker.profileImageUrl ?? null,
        status:    BidStatus.Pending,
        createdAt: new Date(),
      });
      const saved = await bid.save();

      // Status filter: a concurrent accept()/cancel() must not be clobbered
      // back to AwaitingSelection. If we lost that race, roll the bid back.
      const res = await this.requestModel.updateOne(
        {
          _id: dto.serviceRequestId,
          status: { $in: [ServiceStatus.Open, ServiceStatus.AwaitingSelection] },
        },
        { $inc: { bidCount: 1 }, status: ServiceStatus.AwaitingSelection },
      ).exec();
      if (res.matchedCount === 0) {
        await this.bidModel.deleteOne({ _id: saved._id }).exec();
        throw new BadRequestException('Request is no longer accepting bids');
      }

      // ── Realtime push (matches Flutter listeners) ──────────────────────────
      const bidPayload = saved.toObject() as unknown as Record<string, unknown>;
      // → /bids room `request:{id}:bids` → Flutter refreshes the worker bid list.
      this.safeEmit(() =>
        this.bidsGateway.emitBidSubmitted(dto.serviceRequestId, request.userId, bidPayload),
      );
      // → /requests → client tracking screen learns a new bid arrived.
      this.safeEmit(() =>
        this.requestGateway.emitBidReceived(request.userId, dto.serviceRequestId, bidPayload),
      );

      // Inbox + FCM push to the request owner, in the owner's language.
      void this.pushSender.notify(request.userId, {
        type:   'bid_received',
        params: { workerName: worker.name, price: dto.proposedPrice },
        data:   { requestId: dto.serviceRequestId },
      });

      return saved;
    } catch (err) {
      // Compensate: the consumed bid must not be lost to a failed submit.
      if (bidConsumed) {
        await this.usersService.refundBid(uid).catch(() => undefined);
      }
      this.logger.error('BidsService.submit failed', err);
      throw err;
    }
  }

  async findById(id: string): Promise<WorkerBidDocument> {
    try {
      const doc = await this.bidModel.findById(id).exec();
      if (!doc) throw new NotFoundException(`Bid ${id} not found`);
      return doc;
    } catch (err) {
      this.logger.error(`BidsService.findById(${id}) failed`, err);
      throw err;
    }
  }

  async findMany(filters: BidFilters): Promise<WorkerBidDocument[]> {
    try {
      const query: Partial<Record<string, unknown>> = {};
      if (filters.serviceRequestId) query['serviceRequestId'] = filters.serviceRequestId;
      if (filters.workerId)         query['workerId']         = filters.workerId;
      if (filters.status)           query['status']           = filters.status;

      // Filter out expired accepted bids (TTL: 7 days since acceptance).
      query['$or'] = [
        { status: { $ne: BidStatus.Accepted } },
        { expiresAt: { $gt: new Date() } },
      ];

      return this.bidModel
        .find(query)
        .sort({ createdAt: 1 })
        .limit(Math.min(filters.limit ?? 50, 100))
        .exec();
    } catch (err) {
      this.logger.error('BidsService.findMany failed', err);
      throw err;
    }
  }

  async accept(bidId: string, uid: string): Promise<void> {
    try {
      const bid = await this.bidModel.findById(bidId).exec();
      if (!bid) throw new NotFoundException(`Bid ${bidId} not found`);

      const request = await this.requestModel.findById(bid.serviceRequestId).exec();
      if (!request) throw new NotFoundException(`Service request ${bid.serviceRequestId} not found`);
      if (request.userId !== uid) throw new ForbiddenException('Only the request owner can accept a bid');

      // ── Step 1: claim the bid atomically (Pending → Accepted). ──────────────
      // The status filter loses cleanly against a concurrent withdraw(). A bid
      // already Accepted falls through: either an idempotent client retry, or
      // crash-repair for a previous attempt that died before step 2.
      if (bid.status === BidStatus.Pending) {
        const now = new Date();
        const expiresAt = new Date(now.getTime() + BID_TTL_DAYS * 24 * 60 * 60 * 1000);
        const bidClaim = await this.bidModel.updateOne(
          { _id: bidId, status: BidStatus.Pending },
          { status: BidStatus.Accepted, acceptedAt: now, expiresAt },
        ).exec();
        if (bidClaim.matchedCount === 0) {
          throw new BadRequestException('Bid is not pending');
        }
      } else if (bid.status !== BidStatus.Accepted) {
        throw new BadRequestException(`Bid is not pending (status: ${bid.status})`);
      }

      // ── Step 2: claim the request atomically. ────────────────────────────────
      // Only one accept can flip Open/AwaitingSelection → BidSelected; the $or
      // branch lets a retry of THIS bid pass (idempotent / crash-repair).
      const reqClaim = await this.requestModel.updateOne(
        {
          _id: bid.serviceRequestId,
          $or: [
            { status: { $in: [ServiceStatus.Open, ServiceStatus.AwaitingSelection] } },
            { status: ServiceStatus.BidSelected, selectedBidId: bidId },
          ],
        },
        {
          status:        ServiceStatus.BidSelected,
          selectedBidId: bidId,
          workerId:      bid.workerId,
          workerName:    bid.workerName,
          agreedPrice:   bid.proposedPrice,
          bidSelectedAt: new Date(),
        },
      ).exec();

      if (reqClaim.matchedCount === 0) {
        // Lost to a concurrent accept/cancel — roll the bid back and report.
        // ponytail: no transaction (standalone Mongo); a crash between the two
        // claims is self-repairing via the Accepted fall-through above.
        await this.bidModel.updateOne(
          { _id: bidId, status: BidStatus.Accepted },
          { status: BidStatus.Pending, acceptedAt: null },
        ).exec();
        const now = await this.requestModel.findById(bid.serviceRequestId).select('status').lean().exec();
        throw new BadRequestException(
          `Cannot accept bid on request in status: ${(now as { status?: string } | null)?.status ?? 'unknown'}`,
        );
      }

      // ── Step 3: decline the other pending bids and tell their workers. ──────
      const losers = await this.bidModel
        .find({ serviceRequestId: bid.serviceRequestId, _id: { $ne: bidId }, status: BidStatus.Pending })
        .select('workerId')
        .lean()
        .exec();

      await this.bidModel.updateMany(
        { serviceRequestId: bid.serviceRequestId, _id: { $ne: bidId }, status: BidStatus.Pending },
        { status: BidStatus.Declined },
      ).exec();

      // → /requests → both client and assigned worker refresh the request view.
      this.safeEmit(() =>
        this.requestGateway.emitRequestUpdated(bid.serviceRequestId, {
          status:   ServiceStatus.BidSelected,
          workerId: bid.workerId,
        }),
      );
      this.safeEmit(() =>
        this.bidsGateway.emitBidAccepted(bid.serviceRequestId, bidId, bid.workerId),
      );

      // Inbox + FCM push to the winning worker, in the worker's language.
      void this.pushSender.notify(bid.workerId, {
        type: 'bid_accepted',
        data: { requestId: bid.serviceRequestId },
      });

      // Losing bidders: realtime + inbox/push so their my-bids tab doesn't lie.
      for (const loser of losers as Array<{ _id: string; workerId: string }>) {
        this.safeEmit(() =>
          this.bidsGateway.emitBidDeclined(loser.workerId, bid.serviceRequestId, loser._id),
        );
        void this.pushSender.notify(loser.workerId, {
          type: 'bid_declined',
          data: { requestId: bid.serviceRequestId },
        });
      }
    } catch (err) {
      this.logger.error(`BidsService.accept(${bidId}) failed`, err);
      throw err;
    }
  }

  async withdraw(bidId: string, uid: string): Promise<void> {
    try {
      const bid = await this.bidModel.findById(bidId).exec();
      if (!bid) throw new NotFoundException(`Bid ${bidId} not found`);
      if (bid.workerId !== uid) throw new ForbiddenException('You can only withdraw your own bids');

      // Atomic claim: loses cleanly against a concurrent accept()/double-tap.
      const claim = await this.bidModel.updateOne(
        { _id: bidId, status: BidStatus.Pending },
        { status: BidStatus.Withdrawn },
      ).exec();
      if (claim.matchedCount === 0) {
        throw new BadRequestException(`Can only withdraw pending bids (status: ${bid.status})`);
      }

      // $gt guard: a double-decrement race can never push the counter negative.
      await this.requestModel.updateOne(
        { _id: bid.serviceRequestId, bidCount: { $gt: 0 } },
        { $inc: { bidCount: -1 } },
      ).exec();

      // → /requests → client tracking screen refreshes the (now smaller) bid set.
      this.safeEmit(() =>
        this.requestGateway.emitRequestUpdated(bid.serviceRequestId, { bidWithdrawn: true }),
      );
      this.safeEmit(() =>
        this.bidsGateway.emitBidWithdrawn(bid.serviceRequestId, bidId, bid.workerId),
      );
    } catch (err) {
      this.logger.error(`BidsService.withdraw(${bidId}) failed`, err);
      throw err;
    }
  }
}
