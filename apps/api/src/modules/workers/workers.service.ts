// ══════════════════════════════════════════════════════════════════════════════
// WorkersService — Facade over UsersService
//
// PATTERN: Facade / View
//   This service does NOT inject a Mongoose model. It delegates every operation
//   to UsersService with an implicit role='worker' filter.
//
// WHY A FACADE INSTEAD OF REMOVING THE CLASS ENTIRELY?
//   • WorkersController, LocationService, and the gateway already depend on
//     WorkersService by name. The facade lets those callers stay unchanged.
//   • If you later expose a separate `WorkersModule` to a micro-service, this
//     boundary is already in place.
//   • Zero business logic lives here — it is a typed, documented gateway into
//     the role-discriminated subset of the users collection.
//
// ADDING A NEW WORKER OPERATION:
//   1. Add the method to UsersService (with role=worker filter).
//   2. Add a one-line delegation here.
//   3. Done — WorkersController picks it up automatically.
// ══════════════════════════════════════════════════════════════════════════════

import { BadRequestException, Injectable } from '@nestjs/common';
import { UsersService, UserFilters } from '../users/users.service';
import { MediaService }              from '../media/media.service';
import { UserDocument }              from '../../schemas/user.schema';
import { CreateWorkerDto }           from '../../dto/create-worker.dto';
import { UpdateWorkerDto }           from '../../dto/update-worker.dto';

export interface WorkerFilters extends Omit<UserFilters, 'role'> {}

@Injectable()
export class WorkersService {
  constructor(
    private readonly usersService: UsersService,
    private readonly mediaService: MediaService,
  ) {}

  // ── CRUD ──────────────────────────────────────────────────────────────────

  async upsert(dto: CreateWorkerDto): Promise<UserDocument> {
    return this.usersService.upsertWorker(dto);
  }

  async findById(id: string): Promise<UserDocument> {
    return this.usersService.findWorkerById(id);
  }

  async findByIdOrNull(id: string): Promise<UserDocument | null> {
    return this.usersService.findWorkerByIdOrNull(id);
  }

  async findMany(filters: WorkerFilters): Promise<UserDocument[]> {
    return this.usersService.findWorkers(filters);
  }

  async update(id: string, dto: UpdateWorkerDto): Promise<UserDocument> {
    return this.usersService.updateWorker(id, dto);
  }

  // ── Portfolio ─────────────────────────────────────────────────────────────

  /**
   * Set the worker's showcase photos. The two-step split is deliberate: media
   * ownership is MediaService's business (the URL must be an image THIS worker
   * uploaded to OUR Cloudinary account — otherwise the field is an image-host
   * injection into every client's app), the quota is UsersService's.
   *
   * Photos dropped from the list are released on Cloudinary afterwards, so a
   * worker replacing their gallery ten times doesn't leave ten dead objects
   * behind. Best-effort: cleanup failures never fail the save.
   */
  async setPortfolio(id: string, urls: string[]): Promise<UserDocument> {
    const bad = urls.find((u) => !this.mediaService.isOwnPortfolioImage(u, id));
    if (bad) {
      throw new BadRequestException(
        'portfolio must contain only portfolio images uploaded by this worker',
      );
    }
    const removed = await this.usersService.setPortfolio(id, urls);
    for (const url of removed) void this.mediaService.destroyByUrl(url);
    return this.findById(id);
  }

  // ── Status ────────────────────────────────────────────────────────────────

  async updateStatus(id: string, isOnline: boolean): Promise<void> {
    return this.usersService.updateWorkerStatus(id, isOnline);
  }

  // ── Location ──────────────────────────────────────────────────────────────

  async updateLocation(
    id: string,
    latitude: number,
    longitude: number,
    cellId?: string,
    wilayaCode?: number,
    geoHash?: string,
  ): Promise<void> {
    return this.usersService.updateWorkerLocation(id, latitude, longitude, cellId, wilayaCode, geoHash);
  }

  // ── FCM ───────────────────────────────────────────────────────────────────

  async updateFcmToken(id: string, fcmToken: string): Promise<void> {
    return this.usersService.updateWorkerFcmToken(id, fcmToken);
  }

  // ── Rating ────────────────────────────────────────────────────────────────

  /**
   * Apply Bayesian average rating update when a new review comes in.
   * Delegates to UsersService — single authoritative implementation.
   */
  async applyRating(id: string, stars: number): Promise<void> {
    return this.usersService.applyRating(id, stars);
  }
}
