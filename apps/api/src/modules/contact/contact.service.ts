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
import {
  ContactInteraction,
  ContactInteractionDocument,
} from '../../schemas/contact-interaction.schema';
import { ServiceRequest, ServiceRequestDocument } from '../../schemas/service-request.schema';
import { WorkerBid, WorkerBidDocument } from '../../schemas/worker-bid.schema';
import { BidStatus, ServiceStatus } from '../../common/enums';
import { UsersService } from '../users/users.service';
import { BidsGateway } from '../gateway/bids.gateway';
import { ServiceRequestGateway } from '../gateway/service-request.gateway';
import { PushSenderService } from '../notifications/push-sender.service';
import { ResolveContactDto } from './dto/resolve-contact.dto';

const RATEABLE_DAYS = 30;
const REMINDER_HOURS = 48;

/**
 * ContactService — owns the Quote → Contact → Rating spine.
 *
 * acceptQuote() is the ONLY writer of ContactInteraction (idempotent on
 * quoteId). resolve() is the ONLY writer of outcome/stars/tags (write-once).
 * dueReminders() is the read side of the 48h sweep (marking stays with the
 * sweep runner so tests never need timers).
 *
 * The Map/Directory flow NEVER touches this service — visible phone there
 * is intentional and creates no interaction.
 */
@Injectable()
export class ContactService {
  private readonly logger = new Logger(ContactService.name);

  constructor(
    @InjectModel(ContactInteraction.name)
    private readonly contactModel: Model<ContactInteractionDocument>,
    @InjectModel(ServiceRequest.name)
    private readonly requestModel: Model<ServiceRequestDocument>,
    @InjectModel(WorkerBid.name)
    private readonly bidModel: Model<WorkerBidDocument>,
    private readonly usersService: UsersService,
    private readonly bidsGateway: BidsGateway,
    private readonly requestGateway: ServiceRequestGateway,
    private readonly pushSender: PushSenderService,
  ) {}

  private safeEmit(fn: () => void): void {
    try {
      fn();
    } catch (err) {
      this.logger.warn(`Realtime emit failed (non-fatal): ${(err as Error).message}`);
    }
  }

  /**
   * Client accepts a quote — the unlock event.
   *
   * Transaction: claim bid (Pending → Accepted) → claim request
   * (Open/AwaitingSelection → BidSelected) → decline losers →
   * create interaction (idempotent) → notify winner (+ inbox, unmuted) →
   * notify losers (bid_declined). A crash between the claims is
   * self-repairing via the Accepted fall-through (same pattern as before).
   */
  async acceptQuote(bidId: string, uid: string): Promise<{ interactionId: string }> {
    const bid = await this.bidModel.findById(bidId).exec();
    if (!bid) throw new NotFoundException(`Bid ${bidId} not found`);

    const request = await this.requestModel.findById(bid.serviceRequestId).exec();
    if (!request) throw new NotFoundException(`Service request ${bid.serviceRequestId} not found`);
    if (request.userId !== uid) throw new ForbiddenException('Only the request owner can accept a bid');

    // ── Step 1: claim the bid (Pending → Accepted). ──────────────────────
    if (bid.status === BidStatus.Pending) {
      const bidClaim = await this.bidModel.updateOne(
        { _id: bidId, status: BidStatus.Pending },
        { status: BidStatus.Accepted, acceptedAt: new Date() },
      ).exec();
      if (bidClaim.matchedCount === 0) throw new BadRequestException('Bid is not pending');
    } else if (bid.status !== BidStatus.Accepted) {
      throw new BadRequestException(`Bid is not pending (status: ${bid.status})`);
    }

    // ── Step 2: claim the request (Open/AwaitingSelection → BidSelected). ─
    const reqClaim = await this.requestModel.updateOne(
      {
        _id: bid.serviceRequestId,
        $or: [
          { status: { $in: [ServiceStatus.Open, ServiceStatus.AwaitingSelection] } },
          { status: ServiceStatus.BidSelected, selectedBidId: bidId },
        ],
      },
      {
        status: ServiceStatus.BidSelected,
        selectedBidId: bidId,
        workerId: bid.workerId,
        workerName: bid.workerName,
        agreedPrice: bid.proposedPrice,
        bidSelectedAt: new Date(),
      },
    ).exec();

    if (reqClaim.matchedCount === 0) {
      await this.bidModel.updateOne(
        { _id: bidId, status: BidStatus.Accepted },
        { status: BidStatus.Pending, acceptedAt: null },
      ).exec();
      const now = await this.requestModel.findById(bid.serviceRequestId).select('status').lean().exec();
      throw new BadRequestException(
        `Cannot accept bid on request in status: ${(now as { status?: string } | null)?.status ?? 'unknown'}`,
      );
    }

    // ── Step 3: decline the other pending bids and tell their workers. ───
    const losers = await this.bidModel
      .find({ serviceRequestId: bid.serviceRequestId, _id: { $ne: bidId }, status: BidStatus.Pending })
      .select('workerId')
      .lean()
      .exec();

    await this.bidModel.updateMany(
      { serviceRequestId: bid.serviceRequestId, _id: { $ne: bidId }, status: BidStatus.Pending },
      { status: BidStatus.Declined },
    ).exec();

    // ── Step 4: the unlock event — one interaction per quote, ever. ──────
    const now = new Date();
    const rateableUntil = new Date(now.getTime() + RATEABLE_DAYS * 24 * 60 * 60 * 1000);
    await this.contactModel.updateOne(
      { quoteId: bidId },
      {
        $setOnInsert: {
          _id: uuidv4(),
          clientId: request.userId,
          workerId: bid.workerId,
          workerName: bid.workerName,
          serviceType: request.serviceType,
          workerProfileImageUrl: bid.workerProfileImageUrl ?? null,
          source: 'bid_accepted',
          requestId: bid.serviceRequestId,
          quoteId: bidId,
          contactedAt: now,
          rateableUntil,
          outcome: 'pending',
          reminderSent: false,
          resolvedAt: null,
          stars: null,
          tags: [],
          createdAt: now,
        },
      },
      { upsert: true },
    ).exec();
    const interaction = await this.contactModel.findOne({ quoteId: bidId }).select('_id').lean().exec();

    // ── Step 5: realtime + push (fire-and-forget, never break the write). ─
    this.safeEmit(() =>
      this.requestGateway.emitRequestUpdated(bid.serviceRequestId, {
        status: ServiceStatus.BidSelected,
        workerId: bid.workerId,
      }),
    );
    this.safeEmit(() => this.bidsGateway.emitBidAccepted(bid.serviceRequestId, bidId, bid.workerId));

    void this.pushSender.notify(bid.workerId, {
      type: 'bid_accepted',
      data: { requestId: bid.serviceRequestId },
    });

    for (const loser of losers as Array<{ _id: string; workerId: string }>) {
      this.safeEmit(() =>
        this.bidsGateway.emitBidDeclined(loser.workerId, bid.serviceRequestId, loser._id),
      );
      void this.pushSender.notify(loser.workerId, {
        type: 'bid_declined',
        data: { requestId: bid.serviceRequestId },
      });
    }

    return { interactionId: (interaction as { _id: string } | null)?._id ?? '' };
  }

  /**
   * Resolve an interaction — hired (stars+tags → Bayesian rating) or nohire
   * (neutral analytic signal, no rating touch). Write-once: only pending,
   * only by the owning client, only inside the 30d window, never self-rate.
   */
  async resolve(interactionId: string, uid: string, dto: ResolveContactDto): Promise<void> {
    const interaction = await this.contactModel.findById(interactionId).exec();
    if (!interaction) throw new NotFoundException(`Contact interaction ${interactionId} not found`);
    if (interaction.clientId !== uid) throw new ForbiddenException('Only the client can resolve this contact');
    if (interaction.clientId === interaction.workerId) throw new ForbiddenException('Cannot rate yourself');
    if (interaction.outcome !== 'pending') throw new BadRequestException('This contact has already been resolved');
    if (interaction.rateableUntil.getTime() < Date.now()) throw new BadRequestException('Rating window has expired');

    if (dto.outcome === 'hired') {
      if (dto.stars == null) throw new BadRequestException('Stars are required when the job was done');
      const claim = await this.contactModel.updateOne(
        { _id: interactionId, outcome: 'pending' },
        {
          outcome: 'hired',
          stars: dto.stars,
          tags: dto.tags ?? [],
          resolvedAt: new Date(),
        },
      ).exec();
      if (claim.matchedCount === 0) throw new BadRequestException('This contact has already been resolved');

      if (interaction.workerId) {
        try {
          await this.usersService.applyRating(interaction.workerId, dto.stars);
        } catch (err) {
          this.logger.warn(`applyRating(${interaction.workerId}) failed (rating saved): ${(err as Error).message}`);
        }
      }
    } else {
      const claim = await this.contactModel.updateOne(
        { _id: interactionId, outcome: 'pending' },
        { outcome: 'nohire', resolvedAt: new Date() },
      ).exec();
      if (claim.matchedCount === 0) throw new BadRequestException('This contact has already been resolved');
      // nohire touches NOTHING else — no rating, no counter, no rank.
    }

    this.safeEmit(() =>
      this.requestGateway.emitRequestUpdated(interaction.requestId, { rated: true }),
    );
  }

  async findForClient(clientId: string, uid: string, limit = 50): Promise<ContactInteractionDocument[]> {
    if (clientId !== uid) throw new ForbiddenException('You can only list your own contacts');
    return this.contactModel
      .find({ clientId })
      .sort({ createdAt: -1 })
      .limit(Math.min(limit, 100))
      .exec();
  }

  /**
   * Read side of the 48h sweep: pending, unnagged, old enough, not expired.
   * The runner marks reminderSent — this method never writes.
   */
  async dueReminders(now: Date, limit = 200): Promise<ContactInteractionDocument[]> {
    const threshold = new Date(now.getTime() - REMINDER_HOURS * 60 * 60 * 1000);
    return this.contactModel
      .find({
        outcome: 'pending',
        reminderSent: false,
        contactedAt: { $lte: threshold },
        rateableUntil: { $gt: now },
      })
      .sort({ contactedAt: 1 })
      .limit(Math.min(limit, 500))
      .exec();
  }

  async markReminderSent(id: string): Promise<void> {
    await this.contactModel.updateOne({ _id: id }, { reminderSent: true }).exec();
  }
}
