// apps/api/src/modules/service-requests/service-requests.service.ts
//
// BUG 3 FIX — Browse worker vide
//
// PROBLÈME (multi-couches) :
//   1. La service request créée par le client peut avoir wilayaCode: null en
//      base si le GPS n'était pas résolu. La query WHERE wilayaCode = 31 ne
//      matche pas un document avec wilayaCode: null → résultat vide silencieux.
//   2. Le controller Flutter envoie wilayaCode=0 comme sentinelle quand le
//      worker n'a pas de wilayaCode assigné. Sans garde, on chercherait
//      wilayaCode = 0 en base, ce qui retourne toujours [].
//
// SOLUTION :
//   • Si filters.wilayaCode est défini ET non-nul ET non-zéro :
//       query['wilayaCode'] = { $in: [filters.wilayaCode, null] }
//     → capture à la fois les requests correctement géo-tagguées ET celles
//       qui n'ont pas de wilayaCode (créées avant résolution GPS).
//   • Si filters.wilayaCode est 0 (sentinelle Flutter) ou undefined :
//       aucun filtre géographique → retourne toutes les demandes ouvertes
//       correspondant aux autres critères (serviceType, status…).

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
import { ServiceRequest, ServiceRequestDocument } from '../../schemas/service-request.schema';
import { WorkerBid, WorkerBidDocument }           from '../../schemas/worker-bid.schema';
import { CreateServiceRequestDto }  from '../../dto/create-service-request.dto';
import { UpdateServiceRequestDto }  from '../../dto/update-service-request.dto';
import { SubmitRatingDto }          from '../../dto/submit-rating.dto';
import { ServiceStatus, ServicePriority, BidStatus } from '../../common/enums';
import { UsersService }             from '../users/users.service';
import { ServiceRequestGateway }    from '../gateway/service-request.gateway';
import { PushSenderService }        from '../notifications/push-sender.service';
import { VectorIndexService }       from '../../qdrant/vector-index.service';
import { FlywheelService }          from '../ai/services/flywheel.service';

export interface ServiceRequestFilters {
  userId?: string;
  workerId?: string;
  status?: string | string[];
  // Single wilaya, or several (worker browse expands to neighbouring wilayas so
  // a job just across a wilaya border is still visible — mirrors the client-side
  // worker search which already crosses borders).
  wilayaCode?: number | number[];
  serviceType?: string;
  /** Restrict to these request ids — semantic search hydrates Qdrant hits
   *  through this so vector results obey the same query rules as the browse. */
  ids?: string[];
  limit?: number;
}

@Injectable()
export class ServiceRequestsService {
  private readonly logger = new Logger(ServiceRequestsService.name);

  constructor(
    @InjectModel(ServiceRequest.name)
    private readonly requestModel: Model<ServiceRequestDocument>,
    @InjectModel(WorkerBid.name)
    private readonly bidModel: Model<WorkerBidDocument>,
    // UsersService instead of WorkersService — rating applied on unified collection
    private readonly usersService: UsersService,
    private readonly requestGateway: ServiceRequestGateway,
    private readonly pushSender: PushSenderService,
    private readonly vectorIndex: VectorIndexService,
    private readonly flywheel: FlywheelService,
  ) {}

  /**
   * Fire-and-forget realtime notify. A WebSocket failure must never break the
   * REST mutation — the Mongo write is already the source of truth. Flutter
   * treats `request:updated` purely as a signal to refetch via REST.
   */
  private safeEmitUpdated(requestId: string, update: Record<string, unknown>): void {
    try {
      this.requestGateway.emitRequestUpdated(requestId, update);
    } catch (err) {
      this.logger.warn(`Realtime emit failed (non-fatal): ${(err as Error).message}`);
    }
  }

  async create(dto: CreateServiceRequestDto, uid: string): Promise<ServiceRequestDocument> {
    try {
      if (dto.userId !== uid) throw new ForbiddenException('userId must match authenticated user');

      const request = new this.requestModel({
        _id:      uuidv4(),
        ...dto,
        status:   ServiceStatus.Open,
        priority: dto.priority ?? ServicePriority.Normal,
        bidCount: 0,
        mediaUrls: dto.mediaUrls ?? [],
        createdAt: new Date(),
      });
      const saved = await request.save();

      // Semantic-matching index (fire-and-forget — never blocks creation).
      this.vectorIndex.indexRequest(saved);

      // C1 flywheel — the profession the user committed to labels their
      // latest consented AI query (fire-and-forget, consent re-checked by
      // the query itself: only logged queries can be labeled).
      this.flywheel.recordChoice(uid, dto.serviceType);

      // Push the new lead live to workers browsing this wilaya+service room.
      // Minimal payload (no client PII in the broadcast) — it only triggers a
      // refetch over authenticated HTTP. Non-fatal: WS must never break create.
      if (saved.status === ServiceStatus.Open && saved.wilayaCode != null && saved.serviceType) {
        try {
          this.requestGateway.emitNewAvailableRequest(saved.wilayaCode, saved.serviceType, {
            _id: saved._id, wilayaCode: saved.wilayaCode, serviceType: saved.serviceType,
          });
        } catch (err) {
          this.logger.warn(`New-request emit failed (non-fatal): ${(err as Error).message}`);
        }
      }
      return saved;
    } catch (err) {
      this.logger.error('ServiceRequestsService.create failed', err);
      throw err;
    }
  }

  async findById(id: string): Promise<ServiceRequestDocument> {
    const doc = await this.requestModel.findById(id).exec();
    if (!doc) throw new NotFoundException(`Service request ${id} not found`);
    return doc;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // BUG 3 FIX — findMany()
  //
  // AVANT :
  //   if (filters.wilayaCode != null) query['wilayaCode'] = filters.wilayaCode;
  //   → wilayaCode = 31 ne matche jamais un document avec wilayaCode: null
  //   → résultat [] silencieux pour toutes les requests sans géo-tag
  //
  // APRÈS :
  //   wilayaCode = 0  → sentinelle Flutter "pas de filtre géographique"
  //                     → pas de clause wilayaCode dans la query
  //   wilayaCode > 0  → $in: [wilayaCode, null]
  //                     → matche les requests géo-tagguées ET celles sans tag
  // ─────────────────────────────────────────────────────────────────────────
  async findMany(filters: ServiceRequestFilters): Promise<ServiceRequestDocument[]> {
    const query: Partial<Record<string, unknown>> = {};

    if (filters.userId)   query['userId']   = filters.userId;
    if (filters.workerId) query['workerId'] = filters.workerId;
    if (filters.ids?.length) query['_id']   = { $in: filters.ids };

    // BUG 3 FIX : wilayaCode null en base ne matche jamais { wilayaCode: N }.
    // On utilise $in: [...codes, null] pour capturer les deux cas.
    // wilayaCode = 0 est le sentinelle Flutter "worker sans géo-tag" (filtré ci-dessous) :
    // dans ce cas on ne filtre pas par wilaya pour éviter un résultat vide.
    // Un tableau (worker browse) capture aussi les wilayas voisines.
    const codes = (Array.isArray(filters.wilayaCode)
      ? filters.wilayaCode
      : filters.wilayaCode != null ? [filters.wilayaCode] : []
    ).filter((c) => c > 0);
    if (codes.length > 0) {
      query['wilayaCode'] = { $in: [...codes, null] };
    }
    // aucun code valide → aucun filtre géographique

    if (filters.serviceType) query['serviceType'] = filters.serviceType;

    if (filters.status) {
      query['status'] = Array.isArray(filters.status)
        ? { $in: filters.status }
        : filters.status;
    }

    return this.requestModel
      .find(query)
      .sort({ createdAt: -1 })
      .limit(Math.min(filters.limit ?? 50, 100))
      .exec();
  }

  /**
   * Fields a client is allowed to edit on their OWN request via PATCH.
   *
   * Workflow/state fields (status, workerId, workerName, agreedPrice,
   * selectedBidId, bidSelectedAt, completedAt, workerNotes, finalPrice,
   * clientRating, reviewComment, bidCount) are DELIBERATELY excluded — they may
   * only change through the dedicated flow endpoints (accept-bid, start,
   * complete, rate). The Flutter app sends the full model via `toMap()`, so we
   * accept those keys at the DTO layer but silently ignore them here instead of
   * letting a tampered payload corrupt the request's workflow state.
   */
  private static readonly CLIENT_EDITABLE_FIELDS: readonly (keyof UpdateServiceRequestDto)[] = [
    'userName', 'userPhone', 'serviceType', 'title', 'description',
    'scheduledDate', 'scheduledHour', 'scheduledMinute', 'priority',
    'userLatitude', 'userLongitude', 'userAddress', 'mediaUrls',
    'budgetMin', 'budgetMax', 'cellId', 'wilayaCode', 'geoHash',
  ];

  async update(id: string, dto: UpdateServiceRequestDto, uid: string): Promise<ServiceRequestDocument> {
    const existing = await this.requestModel.findById(id).exec();
    if (!existing) throw new NotFoundException(`Service request ${id} not found`);
    if (existing.userId !== uid) throw new ForbiddenException('You can only update your own requests');

    // A finished or cancelled request is immutable.
    if (
      existing.status === ServiceStatus.Completed ||
      existing.status === ServiceStatus.Cancelled
    ) {
      throw new BadRequestException(`Cannot edit a request in status: ${existing.status}`);
    }

    // Whitelist: copy only client-editable detail fields; ignore workflow fields.
    const patch: Partial<Record<string, unknown>> = {};
    for (const key of ServiceRequestsService.CLIENT_EDITABLE_FIELDS) {
      const value = dto[key];
      if (value !== undefined) patch[key] = value;
    }

    if (Object.keys(patch).length === 0) return existing;

    const doc = await this.requestModel
      .findByIdAndUpdate(id, patch, { new: true, runValidators: true })
      .exec();
    if (!doc) throw new NotFoundException(`Service request ${id} not found`);

    this.vectorIndex.indexRequest(doc); // title/description edits re-embed
    this.safeEmitUpdated(id, { updated: true });
    return doc;
  }

  async cancel(id: string, uid: string): Promise<void> {
    const request = await this.requestModel.findById(id).exec();
    if (!request) throw new NotFoundException(`Service request ${id} not found`);
    if (request.userId !== uid) throw new ForbiddenException('You can only cancel your own requests');

    // Atomic claim: the status filter makes double-cancels and races against
    // complete() lose cleanly instead of resurrecting a finished request.
    const claim = await this.requestModel.updateOne(
      { _id: id, status: { $nin: [ServiceStatus.Completed, ServiceStatus.Cancelled] } },
      { status: ServiceStatus.Cancelled },
    ).exec();
    if (claim.matchedCount === 0) {
      throw new BadRequestException(`Cannot cancel a request in status: ${request.status}`);
    }

    // Close out ALL live bids — the accepted one included, so the worker's
    // my-bids view never shows a won job on a cancelled request.
    await this.bidModel
      .updateMany({ serviceRequestId: id, status: { $in: ['pending', 'accepted'] } }, { status: 'declined' })
      .exec();

    this.vectorIndex.removeRequest(id); // cancelled — never matched again
    this.safeEmitUpdated(id, { status: ServiceStatus.Cancelled });

    // The assigned worker (possibly already en route) must hear about it.
    if (request.workerId) {
      void this.pushSender.notify(request.workerId, {
        type: 'job_cancelled',
        data: { requestId: id },
      });
    }
  }

  /**
   * Assigned worker declines a job they won (before starting it). Inverse of
   * BidsService.accept(): the winning bid is marked declined, the assignment
   * fields are cleared and the request reopens for bids — unlike cancel(),
   * the client's request survives and they are notified to pick again.
   */
  async decline(id: string, uid: string): Promise<void> {
    const request = await this.requestModel.findById(id).exec();
    if (!request) throw new NotFoundException(`Service request ${id} not found`);
    if (request.workerId !== uid) {
      throw new ForbiddenException('Only the assigned worker can decline this job');
    }
    if (request.status !== ServiceStatus.BidSelected) {
      throw new BadRequestException(`Cannot decline a job in status: ${request.status}`);
    }

    // Atomic claim first: if a concurrent startJob()/cancel() won, do NOT
    // decline the bid or clear the assignment.
    const claim = await this.requestModel.updateOne(
      { _id: id, workerId: uid, status: ServiceStatus.BidSelected },
      {
        $set:   { status: ServiceStatus.Open },
        $unset: {
          selectedBidId: '',
          workerId:      '',
          workerName:    '',
          agreedPrice:   '',
          bidSelectedAt: '',
        },
      },
    ).exec();
    if (claim.matchedCount === 0) {
      throw new BadRequestException('Job is no longer in a declinable state');
    }

    if (request.selectedBidId) {
      await this.bidModel
        .updateOne({ _id: request.selectedBidId }, { status: 'declined' })
        .exec();
    }

    this.safeEmitUpdated(id, { status: ServiceStatus.Open, declinedBy: uid });

    // Inbox + FCM push to the client (in their language).
    void this.pushSender.notify(request.userId, {
      type: 'job_declined',
      data: { requestId: id },
    });
  }

  async startJob(id: string, uid: string): Promise<void> {
    const request = await this.requestModel.findById(id).exec();
    if (!request) throw new NotFoundException(`Service request ${id} not found`);
    if (request.workerId !== uid) throw new ForbiddenException('Only the assigned worker can start this job');
    if (request.status !== ServiceStatus.BidSelected) {
      throw new BadRequestException(`Cannot start job in status: ${request.status}`);
    }

    // TTL check: bid expires 7 days after acceptance. Auto-decline if stale.
    const bid = await this.bidModel.findOne({
      serviceRequestId: id,
      workerId: uid,
      status: BidStatus.Accepted,
    }).exec();
    if (bid?.expiresAt && bid.expiresAt.getTime() < Date.now()) {
      throw new BadRequestException('Bid has expired (7 days since acceptance). Contact the client to accept a new bid.');
    }

    const claim = await this.requestModel
      .updateOne(
        { _id: id, workerId: uid, status: ServiceStatus.BidSelected },
        { status: ServiceStatus.InProgress, acceptedAt: new Date() },
      )
      .exec();
    if (claim.matchedCount === 0) {
      throw new BadRequestException('Job is no longer in a startable state');
    }

    this.safeEmitUpdated(id, { status: ServiceStatus.InProgress });

    // Inbox + FCM push to the client (in their language) that work has started.
    void this.pushSender.notify(request.userId, {
      type: 'job_started',
      data: { requestId: id },
    });
  }

  async completeJob(id: string, uid: string, workerNotes?: string, finalPrice?: number): Promise<void> {
    const request = await this.requestModel.findById(id).exec();
    if (!request) throw new NotFoundException(`Service request ${id} not found`);
    if (request.workerId !== uid) throw new ForbiddenException('Only the assigned worker can complete this job');
    if (
      request.status !== ServiceStatus.BidSelected &&
      request.status !== ServiceStatus.InProgress
    ) {
      throw new BadRequestException(`Cannot complete job in status: ${request.status}`);
    }

    const patch: Partial<Record<string, unknown>> = {
      status: ServiceStatus.Completed,
      completedAt: new Date(),
    };
    if (workerNotes) patch['workerNotes'] = workerNotes;
    if (finalPrice != null) patch['finalPrice'] = finalPrice;

    const claim = await this.requestModel.updateOne(
      {
        _id: id,
        workerId: uid,
        status: { $in: [ServiceStatus.BidSelected, ServiceStatus.InProgress] },
      },
      patch,
    ).exec();
    if (claim.matchedCount === 0) {
      throw new BadRequestException('Job is no longer in a completable state');
    }

    // Server-side counter — jobsCompleted is not client-writable. Non-fatal:
    // the completion is already committed.
    try {
      await this.usersService.incrementJobsCompleted(uid);
    } catch (err) {
      this.logger.warn(`incrementJobsCompleted(${uid}) failed (completion saved): ${(err as Error).message}`);
    }

    this.vectorIndex.removeRequest(id); // completed — never matched again
    this.safeEmitUpdated(id, { status: ServiceStatus.Completed });

    // Inbox + FCM push to the client (in their language); prompts them to rate.
    void this.pushSender.notify(request.userId, {
      type: 'job_completed',
      data: { requestId: id },
    });
  }

  async submitRating(id: string, uid: string, dto: SubmitRatingDto): Promise<void> {
    const request = await this.requestModel.findById(id).exec();
    if (!request) throw new NotFoundException(`Service request ${id} not found`);
    if (request.userId !== uid) throw new ForbiddenException('Only the client can rate this request');
    if (request.status !== ServiceStatus.Completed) throw new BadRequestException('Can only rate completed jobs');
    if (request.clientRating != null) throw new BadRequestException('This request has already been rated');

    const patch: Partial<Record<string, unknown>> = { clientRating: dto.stars };
    if (dto.comment) patch['reviewComment'] = dto.comment;

    // clientRating:null filter makes the once-only guard atomic — a concurrent
    // double-submit matches zero documents instead of double-counting.
    const claim = await this.requestModel
      .updateOne({ _id: id, status: ServiceStatus.Completed, clientRating: null }, patch)
      .exec();
    if (claim.matchedCount === 0) {
      throw new BadRequestException('This request has already been rated');
    }

    // Apply Bayesian rating on the unified users collection. Non-fatal: the
    // rating is committed above; a vanished worker must not 500 the client.
    if (request.workerId) {
      try {
        await this.usersService.applyRating(request.workerId, dto.stars);
      } catch (err) {
        this.logger.warn(`applyRating(${request.workerId}) failed (rating saved): ${(err as Error).message}`);
      }
    }

    this.safeEmitUpdated(id, { rated: true });
  }
}
