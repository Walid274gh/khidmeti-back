// ══════════════════════════════════════════════════════════════════════════════
// AdminService
//
// Powers the web admin dashboard. Injects the Mongoose models directly (all are
// exported by the @Global DatabaseModule) rather than reusing the app-facing
// services, because those return un-paginated arrays and enforce end-user
// ownership rules that do not apply to an admin. Reads are paginated + searchable;
// mutations are the moderation actions the app never performs (ban, verify,
// role change, profession CRUD, broadcast).
//
// NOTE: the existing app services (UsersService, ServiceRequestsService, …) are
// intentionally left untouched so no client/worker flow changes behaviour.
// ══════════════════════════════════════════════════════════════════════════════

import {
  BadRequestException,
  ConflictException,
  Injectable,
  Logger,
  NotFoundException,
} from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { FilterQuery, Model } from 'mongoose';
import { User, UserDocument, UserRole } from '../../schemas/user.schema';
import {
  ServiceRequest,
  ServiceRequestDocument,
} from '../../schemas/service-request.schema';
import { WorkerBid, WorkerBidDocument } from '../../schemas/worker-bid.schema';
import { Profession, ProfessionDocument } from '../../schemas/profession.schema';
import { Notification, NotificationDocument } from '../../schemas/notification.schema';
import { BidStatus, ServiceStatus } from '../../common/enums';
import { blindIndex } from '../../common/crypto/field-crypto';
import { decryptLean, USER_ENCRYPTED_FIELDS } from '../../common/crypto/encrypted-fields.plugin';
import { PushSenderService } from '../notifications/push-sender.service';
import { PinGateService } from '../auth/pin-gate.service';
import { VectorIndexService } from '../../qdrant/vector-index.service';
import {
  BroadcastDto,
  ListBidsQueryDto,
  ListRequestsQueryDto,
  ListUsersQueryDto,
  ListWorkersQueryDto,
  PaginationQueryDto,
  UpdateUserAdminDto,
} from './dto/admin-query.dto';
import {
  CreateProfessionDto,
  UpdateProfessionDto,
} from './dto/profession-admin.dto';

export interface Paginated<T> {
  items: T[];
  total: number;
  page: number;
  limit: number;
  pages: number;
}

@Injectable()
export class AdminService {
  private readonly logger = new Logger(AdminService.name);

  constructor(
    @InjectModel(User.name)
    private readonly userModel: Model<UserDocument>,
    @InjectModel(ServiceRequest.name)
    private readonly requestModel: Model<ServiceRequestDocument>,
    @InjectModel(WorkerBid.name)
    private readonly bidModel: Model<WorkerBidDocument>,
    @InjectModel(Profession.name)
    private readonly professionModel: Model<ProfessionDocument>,
    @InjectModel(Notification.name)
    private readonly notificationModel: Model<NotificationDocument>,
    private readonly pushSender: PushSenderService,
    private readonly pinGate: PinGateService,
    private readonly vectorIndex: VectorIndexService,
  ) {}

  // ── Helpers ────────────────────────────────────────────────────────────────

  private paginate(q: PaginationQueryDto): { page: number; limit: number; skip: number } {
    const page = Math.max(1, q.page ?? 1);
    const limit = Math.min(100, Math.max(1, q.limit ?? 20));
    return { page, limit, skip: (page - 1) * limit };
  }

  private sortSpec(sort?: string, fallbackField = 'createdAt'): Record<string, 1 | -1> {
    if (!sort) return { [fallbackField]: -1 };
    const [field, dir] = sort.split(':');
    return { [field || fallbackField]: dir === 'asc' ? 1 : -1 };
  }

  private wrap<T>(items: T[], total: number, page: number, limit: number): Paginated<T> {
    return { items, total, page, limit, pages: Math.max(1, Math.ceil(total / limit)) };
  }

  // ── Dashboard stats ──────────────────────────────────────────────────────────

  async getStats(): Promise<Record<string, unknown>> {
    const [
      totalUsers,
      totalClients,
      totalWorkers,
      onlineWorkers,
      bannedUsers,
      verifiedWorkers,
      totalRequests,
      totalBids,
      requestsByStatus,
      requestsPerDay,
      topProfessions,
    ] = await Promise.all([
      this.userModel.countDocuments().exec(),
      this.userModel.countDocuments({ role: UserRole.Client }).exec(),
      this.userModel.countDocuments({ role: UserRole.Worker }).exec(),
      this.userModel.countDocuments({ role: UserRole.Worker, isOnline: true }).exec(),
      this.userModel.countDocuments({ isBanned: true }).exec(),
      this.userModel.countDocuments({ role: UserRole.Worker, isVerified: true }).exec(),
      this.requestModel.countDocuments().exec(),
      this.bidModel.countDocuments().exec(),
      this.requestModel.aggregate([
        { $group: { _id: '$status', count: { $sum: 1 } } },
      ]),
      this.requestModel.aggregate([
        { $match: { createdAt: { $gte: new Date(Date.now() - 30 * 864e5) } } },
        {
          $group: {
            _id: { $dateToString: { format: '%Y-%m-%d', date: '$createdAt' } },
            count: { $sum: 1 },
          },
        },
        { $sort: { _id: 1 } },
      ]),
      this.userModel.aggregate([
        { $match: { role: UserRole.Worker, profession: { $ne: null } } },
        { $group: { _id: '$profession', count: { $sum: 1 } } },
        { $sort: { count: -1 } },
        { $limit: 8 },
      ]),
    ]);

    const statusMap: Record<string, number> = {};
    for (const row of requestsByStatus as Array<{ _id: string; count: number }>) {
      statusMap[row._id] = row.count;
    }

    return {
      users: { total: totalUsers, clients: totalClients, workers: totalWorkers, banned: bannedUsers },
      workers: { total: totalWorkers, online: onlineWorkers, verified: verifiedWorkers },
      requests: { total: totalRequests, byStatus: statusMap },
      bids: { total: totalBids },
      charts: {
        requestsPerDay: (requestsPerDay as Array<{ _id: string; count: number }>).map((r) => ({
          date: r._id,
          count: r.count,
        })),
        topProfessions: (topProfessions as Array<{ _id: string; count: number }>).map((r) => ({
          profession: r._id,
          count: r.count,
        })),
      },
    };
  }

  // ── Users ──────────────────────────────────────────────────────────────────

  async listUsers(q: ListUsersQueryDto): Promise<Paginated<UserDocument>> {
    const { page, limit, skip } = this.paginate(q);
    const filter: FilterQuery<UserDocument> = {};
    if (q.role) filter.role = q.role;
    if (q.isBanned !== undefined) filter.isBanned = q.isBanned;
    if (q.verificationStatus) {
      // 'approved' is stored as '' (the default). pending/rejected match literally.
      filter.verificationStatus = q.verificationStatus === 'approved' ? '' : q.verificationStatus;
    }
    if (q.search?.trim()) {
      const term = q.search.trim();
      const rx = new RegExp(this.escape(term), 'i');
      // email/phone are encrypted → substring search is impossible. Match the
      // blind index for an EXACT email/phone; name stays substring-searchable.
      filter.$or = [
        { name: rx },
        { emailBidx: blindIndex(term) },
        { phoneNumberBidx: blindIndex(term) },
        { _id: term },
      ];
    }

    const [items, total] = await Promise.all([
      this.userModel.find(filter).sort(this.sortSpec(q.sort, 'lastUpdated')).skip(skip).limit(limit).lean().exec(),
      this.userModel.countDocuments(filter).exec(),
    ]);
    items.forEach((u) => decryptLean(u as Record<string, unknown>, [...USER_ENCRYPTED_FIELDS]));
    return this.wrap(items as unknown as UserDocument[], total, page, limit);
  }

  async getUser(id: string): Promise<UserDocument> {
    const doc = await this.userModel.findById(id).lean().exec();
    if (!doc) throw new NotFoundException(`User ${id} not found`);
    return decryptLean(doc as Record<string, unknown>, [...USER_ENCRYPTED_FIELDS]) as unknown as UserDocument;
  }

  async updateUser(id: string, dto: UpdateUserAdminDto): Promise<UserDocument> {
    const doc = await this.userModel
      .findByIdAndUpdate(id, { $set: { ...dto, lastUpdated: new Date() } }, { new: true })
      .lean()
      .exec();
    if (!doc) throw new NotFoundException(`User ${id} not found`);
    return decryptLean(doc as Record<string, unknown>, [...USER_ENCRYPTED_FIELDS]) as unknown as UserDocument;
  }

  async setRole(id: string, role: UserRole): Promise<UserDocument> {
    const doc = await this.userModel
      .findByIdAndUpdate(id, { $set: { role, lastUpdated: new Date() } }, { new: true })
      .lean()
      .exec();
    if (!doc) throw new NotFoundException(`User ${id} not found`);
    return decryptLean(doc as Record<string, unknown>, [...USER_ENCRYPTED_FIELDS]) as unknown as UserDocument;
  }

  async setBan(id: string, isBanned: boolean): Promise<UserDocument> {
    const doc = await this.userModel
      .findByIdAndUpdate(id, { $set: { isBanned, lastUpdated: new Date() } }, { new: true })
      .lean()
      .exec();
    if (!doc) throw new NotFoundException(`User ${id} not found`);
    return decryptLean(doc as Record<string, unknown>, [...USER_ENCRYPTED_FIELDS]) as unknown as UserDocument;
  }

  /**
   * Approve or reject a submitted verification-document set. Approval clears the
   * status to '' (the auth gate stops blocking) and marks the worker verified;
   * rejection stores the note the mobile user sees. Either way we bust the PIN-
   * gate cache (which also caches verificationStatus) so the change takes effect
   * on the account's very next request, and push a localized notification.
   */
  async setVerification(id: string, status: 'approved' | 'rejected', note = ''): Promise<UserDocument> {
    const set: Record<string, unknown> =
      status === 'approved'
        ? { verificationStatus: '', verificationNote: '', isVerified: true, lastUpdated: new Date() }
        : { verificationStatus: 'rejected', verificationNote: note, lastUpdated: new Date() };

    const doc = await this.userModel
      .findByIdAndUpdate(id, { $set: set }, { new: true })
      .lean()
      .exec();
    if (!doc) throw new NotFoundException(`User ${id} not found`);

    // The gate reads verificationStatus from PinGateService's cache — clear it
    // so the account isn't blocked (or freed) up to CACHE_TTL_MS late.
    this.pinGate.invalidate(id);

    void this.pushSender.notify(id, {
      type: status === 'approved' ? 'verification_approved' : 'verification_rejected',
      params: { note },
    });
    this.logger.log(`Verification ${status} for uid=${id}`);

    return decryptLean(doc as Record<string, unknown>, [...USER_ENCRYPTED_FIELDS]) as unknown as UserDocument;
  }

  async deleteUser(id: string): Promise<{ deleted: true }> {
    const res = await this.userModel.deleteOne({ _id: id }).exec();
    if (res.deletedCount === 0) throw new NotFoundException(`User ${id} not found`);

    // Their vector must not keep surfacing in semantic matching (idempotent
    // for clients — deleting a point that never existed is a no-op).
    this.vectorIndex.removeWorker(id);

    // The cascade below cancels their open requests — capture the ids first
    // so the matching index drops them too (fire-and-forget).
    const openRequestIds = (await this.requestModel
      .find({ userId: id, status: { $nin: [ServiceStatus.Completed, ServiceStatus.Cancelled] } })
      .select('_id')
      .lean()
      .exec()) as unknown as Array<{ _id: string }>;
    for (const r of openRequestIds) this.vectorIndex.removeRequest(r._id);

    // Cascade: close everything keyed on the uid so no live request/bid ever
    // points at a nonexistent user (a deleted worker's pending bid must not
    // stay acceptable). Notifications are just noise — drop them.
    await Promise.all([
      // Their open client requests: nobody can accept/manage them anymore.
      this.requestModel.updateMany(
        { userId: id, status: { $nin: [ServiceStatus.Completed, ServiceStatus.Cancelled] } },
        { status: ServiceStatus.Cancelled },
      ).exec(),
      // Their live worker bids.
      this.bidModel.updateMany(
        { workerId: id, status: BidStatus.Pending },
        { status: BidStatus.Withdrawn },
      ).exec(),
      this.notificationModel.deleteMany({ userId: id }).exec(),
    ]);

    // Requests where they were the assigned worker: unassign and reopen.
    await this.requestModel.updateMany(
      { workerId: id, status: ServiceStatus.BidSelected },
      {
        $set:   { status: ServiceStatus.Open },
        $unset: { selectedBidId: '', workerId: '', workerName: '', agreedPrice: '', bidSelectedAt: '' },
      },
    ).exec();

    return { deleted: true };
  }

  // ── Workers ────────────────────────────────────────────────────────────────

  async listWorkers(q: ListWorkersQueryDto): Promise<Paginated<UserDocument>> {
    const { page, limit, skip } = this.paginate(q);
    const filter: FilterQuery<UserDocument> = { role: UserRole.Worker };
    if (q.profession) filter.profession = q.profession;
    if (q.wilayaCode !== undefined) filter.wilayaCode = q.wilayaCode;
    if (q.isOnline !== undefined) filter.isOnline = q.isOnline;
    if (q.isVerified !== undefined) filter.isVerified = q.isVerified;
    if (q.search?.trim()) {
      const term = q.search.trim();
      const rx = new RegExp(this.escape(term), 'i');
      // See listUsers: encrypted email/phone → exact blind-index match only.
      filter.$or = [
        { name: rx },
        { emailBidx: blindIndex(term) },
        { phoneNumberBidx: blindIndex(term) },
      ];
    }

    const [items, total] = await Promise.all([
      this.userModel.find(filter).sort(this.sortSpec(q.sort, 'averageRating')).skip(skip).limit(limit).lean().exec(),
      this.userModel.countDocuments(filter).exec(),
    ]);
    items.forEach((u) => decryptLean(u as Record<string, unknown>, [...USER_ENCRYPTED_FIELDS]));
    return this.wrap(items as unknown as UserDocument[], total, page, limit);
  }

  async setVerified(id: string, isVerified: boolean): Promise<UserDocument> {
    const doc = await this.userModel
      .findOneAndUpdate({ _id: id, role: UserRole.Worker }, { $set: { isVerified } }, { new: true })
      .lean()
      .exec();
    if (!doc) throw new NotFoundException(`Worker ${id} not found`);
    return decryptLean(doc as Record<string, unknown>, [...USER_ENCRYPTED_FIELDS]) as unknown as UserDocument;
  }

  async setOnline(id: string, isOnline: boolean): Promise<UserDocument> {
    const doc = await this.userModel
      .findOneAndUpdate(
        { _id: id, role: UserRole.Worker },
        { $set: { isOnline, lastActiveAt: isOnline ? null : new Date() } },
        { new: true },
      )
      .lean()
      .exec();
    if (!doc) throw new NotFoundException(`Worker ${id} not found`);
    return decryptLean(doc as Record<string, unknown>, [...USER_ENCRYPTED_FIELDS]) as unknown as UserDocument;
  }

  // ── Service requests ─────────────────────────────────────────────────────────

  async listRequests(q: ListRequestsQueryDto): Promise<Paginated<ServiceRequestDocument>> {
    const { page, limit, skip } = this.paginate(q);
    const filter: FilterQuery<ServiceRequestDocument> = {};
    if (q.status) filter.status = q.status;
    if (q.wilayaCode !== undefined) filter.wilayaCode = q.wilayaCode;
    if (q.search?.trim()) {
      const rx = new RegExp(this.escape(q.search.trim()), 'i');
      filter.$or = [{ title: rx }, { userName: rx }, { serviceType: rx }];
    }

    const [items, total] = await Promise.all([
      this.requestModel.find(filter).sort(this.sortSpec(q.sort)).skip(skip).limit(limit).lean().exec(),
      this.requestModel.countDocuments(filter).exec(),
    ]);
    return this.wrap(items as unknown as ServiceRequestDocument[], total, page, limit);
  }

  async getRequest(id: string): Promise<ServiceRequestDocument> {
    const doc = await this.requestModel.findById(id).lean().exec();
    if (!doc) throw new NotFoundException(`Request ${id} not found`);
    return doc as unknown as ServiceRequestDocument;
  }

  async cancelRequest(id: string): Promise<ServiceRequestDocument> {
    const doc = await this.requestModel
      .findByIdAndUpdate(id, { $set: { status: ServiceStatus.Cancelled } }, { new: true })
      .lean()
      .exec();
    if (!doc) throw new NotFoundException(`Request ${id} not found`);
    this.vectorIndex.removeRequest(id); // cancelled — never matched again
    return doc as unknown as ServiceRequestDocument;
  }

  // ── Bids ─────────────────────────────────────────────────────────────────────

  async listBids(q: ListBidsQueryDto): Promise<Paginated<WorkerBidDocument>> {
    const { page, limit, skip } = this.paginate(q);
    const filter: FilterQuery<WorkerBidDocument> = {};
    if (q.status) filter.status = q.status;
    if (q.workerId) filter.workerId = q.workerId;
    if (q.serviceRequestId) filter.serviceRequestId = q.serviceRequestId;
    if (q.search?.trim()) {
      const rx = new RegExp(this.escape(q.search.trim()), 'i');
      filter.$or = [{ workerName: rx }];
    }

    const [items, total] = await Promise.all([
      this.bidModel.find(filter).sort(this.sortSpec(q.sort)).skip(skip).limit(limit).lean().exec(),
      this.bidModel.countDocuments(filter).exec(),
    ]);
    return this.wrap(items as unknown as WorkerBidDocument[], total, page, limit);
  }

  // ── Professions (full CRUD) ──────────────────────────────────────────────────

  async listProfessions(): Promise<ProfessionDocument[]> {
    return this.professionModel.find().sort({ sortOrder: 1 }).lean().exec() as unknown as ProfessionDocument[];
  }

  async createProfession(dto: CreateProfessionDto): Promise<ProfessionDocument> {
    const exists = await this.professionModel.findOne({ key: dto.key }).lean().exec();
    if (exists) throw new ConflictException(`Profession key '${dto.key}' already exists`);
    return this.professionModel.create(dto as unknown as Profession);
  }

  async updateProfession(key: string, dto: UpdateProfessionDto): Promise<ProfessionDocument> {
    const doc = await this.professionModel
      .findOneAndUpdate({ key }, { $set: dto }, { new: true })
      .lean()
      .exec();
    if (!doc) throw new NotFoundException(`Profession '${key}' not found`);
    return doc as unknown as ProfessionDocument;
  }

  async deleteProfession(key: string): Promise<{ deleted: true }> {
    const res = await this.professionModel.deleteOne({ key }).exec();
    if (res.deletedCount === 0) throw new NotFoundException(`Profession '${key}' not found`);
    return { deleted: true };
  }

  // ── Broadcast ────────────────────────────────────────────────────────────────

  async broadcast(dto: BroadcastDto): Promise<{ recipients: number }> {
    const filter: FilterQuery<UserDocument> = {};
    switch (dto.audience) {
      case 'clients':
        filter.role = UserRole.Client;
        break;
      case 'workers':
        filter.role = UserRole.Worker;
        break;
      case 'wilaya':
        if (dto.wilayaCode === undefined) {
          throw new BadRequestException('wilayaCode is required for audience=wilaya');
        }
        filter.wilayaCode = dto.wilayaCode;
        break;
      case 'all':
      default:
        break;
    }
    filter.isBanned = { $ne: true } as unknown as boolean;

    const recipients = await this.userModel.find(filter).select('_id').lean().exec();

    // Chunked fan-out: notifyRaw never throws, but 50k un-awaited sends at once
    // would exhaust sockets/heap. 50 in flight at a time, fire-and-forget as a
    // whole so the HTTP response returns immediately.
    void (async () => {
      const CHUNK = 50;
      for (let i = 0; i < recipients.length; i += CHUNK) {
        await Promise.all(
          (recipients as Array<{ _id: string }>).slice(i, i + CHUNK).map((r) =>
            this.pushSender.notifyRaw(r._id, {
              title: dto.title,
              body: dto.body,
              type: 'admin_broadcast',
            }),
          ),
        );
      }
    })();

    this.logger.log(`Broadcast '${dto.title}' queued to ${recipients.length} recipient(s)`);
    return { recipients: recipients.length };
  }

  // ── util ─────────────────────────────────────────────────────────────────────

  /** Escape user-supplied text before embedding in a RegExp (ReDoS / injection safety). */
  private escape(input: string): string {
    return input.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }
}
