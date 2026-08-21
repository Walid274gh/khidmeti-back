// apps/api/src/modules/gateway/worker-location.gateway.ts
//
// FIX 1 — Repeated "Worker not found" errors in logs
//
// ROOT CAUSE:
//   On every WebSocket connection to /workers namespace, handleConnection()
//   performs a MongoDB findOne({ _id: uid, role: 'worker' }).
//   When a CLIENT connects to this namespace (which happens because Flutter
//   uses the same socket URL for location updates), the query returns null,
//   socket.data.isWorker = false, and "Worker not found" is logged as WARNING.
//   This is NOT an error — it's expected behaviour — but the repeated logs
//   create noise that hides real errors.
//
// FIX 1:
//   1. Cache the worker lookup result per UID in a Map to avoid repeated
//      DB queries when the same worker reconnects.
//   2. Downgrade "non-worker connected" from WARN to DEBUG.
//
// FIX 2 — Map shows no workers despite seeded data
//
// ROOT CAUSE:
//   subscribe:wilaya only called socket.join(room) — no initial state was
//   emitted. Seeded workers have fake UIDs and never connect via WebSocket,
//   so they never emit worker:location events. Flutter received zero events
//   after subscribing and the map stayed empty even though MongoDB had 10
//   online workers.
//
// FIX 2 — workers:snapshot on subscribe:wilaya:
//   After joining the room, query MongoDB for current online workers in
//   that wilaya and emit a `workers:snapshot` event directly to the
//   subscribing socket. Flutter handles `workers:snapshot` to populate
//   the initial map markers, then `worker:location` / `worker:status` for
//   live diffs. Non-fatal: snapshot errors are logged at WARN level and
//   do not disconnect the socket.

import {
  WebSocketGateway,
  WebSocketServer,
  SubscribeMessage,
  MessageBody,
  ConnectedSocket,
  OnGatewayConnection,
  OnGatewayDisconnect,
} from '@nestjs/websockets';
import { Logger } from '@nestjs/common';
import { Server, Socket } from 'socket.io';
import * as admin from 'firebase-admin';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { User, UserDocument, UserRole } from '../../schemas/user.schema';
import { PinGateService } from '../auth/pin-gate.service';

interface AuthenticatedSocket extends Socket {
  data: { uid: string; isWorker: boolean; wilayaCode?: number };
}

interface LocationPayload { lat: number; lng: number; }
interface StatusPayload   { isOnline: boolean; }

// ── Worker profile cache ───────────────────────────────────────────────────────
// Avoids one MongoDB query per reconnection. TTL: 5 minutes.

interface CachedWorkerProfile {
  isWorker:   boolean;
  wilayaCode: number | undefined;
  profession: string | undefined;
  cachedAt:   number;
}

const PROFILE_CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes

@WebSocketGateway({
  namespace: '/workers',
  cors: { origin: '*', credentials: false },
  transports: ['websocket', 'polling'],
})
export class WorkerLocationGateway implements OnGatewayConnection, OnGatewayDisconnect {
  @WebSocketServer() private readonly server!: Server;
  private readonly logger = new Logger(WorkerLocationGateway.name);

  private readonly profileCache = new Map<string, CachedWorkerProfile>();

  constructor(
    @InjectModel(User.name)
    private readonly userModel: Model<UserDocument>,
    private readonly pinGate: PinGateService,
  ) {}

  // ── Connection lifecycle ────────────────────────────────────────────────────

  async handleConnection(socket: AuthenticatedSocket): Promise<void> {
    try {
      const token =
        socket.handshake.auth?.['token'] as string | undefined ??
        (socket.handshake.headers['authorization'] as string | undefined)?.replace('Bearer ', '');

      if (!token) {
        this.logger.debug(`[WS workers] Rejected unauthenticated socket ${socket.id}`);
        socket.disconnect(true);
        return;
      }

      const decoded = await admin.auth().verifyIdToken(token);
      const uid     = decoded.uid;

      // Account-PIN device gate — same rule as FirebaseAuthGuard for HTTP.
      const deviceId = socket.handshake.auth?.['deviceId'] as string | undefined;
      if (!(await this.pinGate.isDeviceAllowed(uid, deviceId))) {
        this.logger.warn(`[WS workers] PIN_REQUIRED — rejected ${uid} (socket ${socket.id})`);
        socket.disconnect(true);
        return;
      }

      // Document-approval gate — un-approved accounts get no realtime either.
      if (!(await this.pinGate.isApproved(uid))) {
        this.logger.warn(`[WS workers] APPROVAL_PENDING — rejected ${uid} (socket ${socket.id})`);
        socket.disconnect(true);
        return;
      }

      socket.data.uid = uid;

      const profile = await this.getWorkerProfile(uid);

      socket.data.isWorker   = profile.isWorker;
      socket.data.wilayaCode = profile.wilayaCode;

      if (profile.isWorker) {
        await socket.join(`worker:${uid}`);
        if (profile.wilayaCode) {
          await socket.join(`wilaya:${profile.wilayaCode}`);
        }
        this.logger.log(`[WS workers] Worker ${uid} connected (${socket.id})`);
      } else {
        // FIX 1: Downgraded from WARN to DEBUG — clients connecting to /workers
        // is expected behaviour (they subscribe to worker locations on the map).
        this.logger.debug(`[WS workers] Viewer ${uid} connected (${socket.id})`);
      }
    } catch (err) {
      this.logger.warn(`[WS workers] Auth failure on socket ${socket.id}: ${err}`);
      socket.disconnect(true);
    }
  }

  handleDisconnect(socket: AuthenticatedSocket): void {
    this.logger.debug(
      `[WS workers] Socket ${socket.id} (uid=${socket.data?.uid ?? 'unknown'}) disconnected`,
    );
  }

  // ── Worker → Server events ──────────────────────────────────────────────────

  @SubscribeMessage('worker:update_location')
  async handleUpdateLocation(
    @ConnectedSocket() socket: AuthenticatedSocket,
    @MessageBody() payload: LocationPayload,
  ): Promise<void> {
    if (!socket.data?.isWorker) return;
    if (!payload || typeof payload !== 'object') return;
    const { lat, lng } = payload;
    if (typeof lat !== 'number' || typeof lng !== 'number') return;
    if (lat < -90 || lat > 90 || lng < -180 || lng > 180) return;

    const workerId = socket.data.uid;

    this.userModel
      .updateOne(
        { _id: workerId, role: UserRole.Worker },
        { latitude: lat, longitude: lng, lastUpdated: new Date() },
      )
      .exec()
      .catch((e: unknown) => this.logger.error('Location persist failed', e));

    const event = { workerId, lat, lng, ts: Date.now() };
    if (socket.data.wilayaCode) {
      this.server.to(`wilaya:${socket.data.wilayaCode}`).emit('worker:location', event);
    }
    this.server.to(`worker:${workerId}`).emit('worker:location', event);
  }

  @SubscribeMessage('worker:set_status')
  async handleSetStatus(
    @ConnectedSocket() socket: AuthenticatedSocket,
    @MessageBody() payload: StatusPayload,
  ): Promise<void> {
    if (!socket.data?.isWorker) return;
    if (!payload || typeof payload !== 'object') return;
    const { isOnline } = payload;
    if (typeof isOnline !== 'boolean') return;

    const workerId = socket.data.uid;

    try {
      await this.userModel
        .updateOne(
          { _id: workerId, role: UserRole.Worker },
          {
            isOnline,
            lastUpdated: new Date(),
            ...(isOnline ? {} : { lastActiveAt: new Date() }),
          },
        )
        .exec();
    } catch (e) {
      this.logger.error('Status persist failed', e);
      return; // don't broadcast a state we failed to persist
    }

    // Bust profile cache so next connection reflects updated isOnline
    this.profileCache.delete(workerId);

    const event = { workerId, isOnline, ts: Date.now() };
    if (socket.data.wilayaCode) {
      this.server.to(`wilaya:${socket.data.wilayaCode}`).emit('worker:status', event);
    }
    this.server.to(`worker:${workerId}`).emit('worker:status', event);
  }

  // ── Client → Server: room subscriptions ────────────────────────────────────

  /**
   * Subscribe to online workers in a wilaya.
   *
   * FIX 2 — workers:snapshot:
   *   After joining the room, emit the current snapshot of online workers
   *   directly to the subscribing socket so the Flutter map can populate
   *   initial markers without waiting for live events.
   *
   *   Event shape: {
   *     wilayaCode: number,
   *     workers: Array<{
   *       workerId: string, lat: number, lng: number,
   *       profession: string|null, averageRating: number,
   *       isOnline: true, ts: number
   *     }>
   *   }
   *
   *   Non-fatal: snapshot errors are caught and logged — they never
   *   disconnect the socket or block the room join.
   */
  @SubscribeMessage('subscribe:wilaya')
  async handleSubscribeWilaya(
    @ConnectedSocket() socket: AuthenticatedSocket,
    @MessageBody() payload: { wilayaCode: number },
  ): Promise<void> {
    if (!payload?.wilayaCode || typeof payload.wilayaCode !== 'number') return;

    await socket.join(`wilaya:${payload.wilayaCode}`);

    // ── FIX 2: emit current snapshot to the subscribing socket ──────────────
    try {
      const workers = await this.userModel
        .find({
          role:      UserRole.Worker,
          isOnline:  true,
          wilayaCode: payload.wilayaCode,
          latitude:  { $ne: null },
          longitude: { $ne: null },
        })
        .select('_id latitude longitude profession averageRating wilayaCode isOnline')
        .lean()
        .exec();

      socket.emit('workers:snapshot', {
        wilayaCode: payload.wilayaCode,
        workers: workers.map((w: any) => ({
          workerId:      String(w._id),
          lat:           w.latitude  as number,
          lng:           w.longitude as number,
          profession:    (w.profession as string | null) ?? null,
          averageRating: (w.averageRating as number) ?? 0,
          isOnline:      true,
          ts:            Date.now(),
        })),
      });

      this.logger.debug(
        `[WS workers] snapshot → socket ${socket.id} ` +
        `wilaya=${payload.wilayaCode} count=${workers.length}`,
      );
    } catch (err) {
      // Non-fatal — the client is still in the room and will receive live events.
      this.logger.warn(
        `[WS workers] snapshot failed for wilaya=${payload.wilayaCode}: ` +
        `${(err as Error).message}`,
      );
    }
  }

  @SubscribeMessage('subscribe:worker')
  async handleSubscribeWorker(
    @ConnectedSocket() socket: AuthenticatedSocket,
    @MessageBody() payload: { workerId: string },
  ): Promise<void> {
    if (!payload?.workerId || typeof payload.workerId !== 'string') return;
    await socket.join(`worker:${payload.workerId}`);
  }

  // ── Server-initiated helpers ─────────────────────────────────────────────────

  emitWorkerLocation(workerId: string, lat: number, lng: number, wilayaCode?: number): void {
    const event = { workerId, lat, lng, ts: Date.now() };
    if (wilayaCode) this.server.to(`wilaya:${wilayaCode}`).emit('worker:location', event);
    this.server.to(`worker:${workerId}`).emit('worker:location', event);
  }

  emitWorkerStatus(workerId: string, isOnline: boolean, wilayaCode?: number): void {
    const event = { workerId, isOnline, ts: Date.now() };
    if (wilayaCode) this.server.to(`wilaya:${wilayaCode}`).emit('worker:status', event);
    this.server.to(`worker:${workerId}`).emit('worker:status', event);
  }

  // ── Profile cache ────────────────────────────────────────────────────────────

  private async getWorkerProfile(uid: string): Promise<CachedWorkerProfile> {
    const cached = this.profileCache.get(uid);
    if (cached && Date.now() - cached.cachedAt < PROFILE_CACHE_TTL_MS) {
      return cached;
    }

    const user = await this.userModel
      .findOne({ _id: uid, role: UserRole.Worker })
      .select('wilayaCode profession isOnline')
      .lean()
      .exec();

    const profile: CachedWorkerProfile = {
      isWorker:   !!user,
      wilayaCode: user ? (user as any).wilayaCode ?? undefined : undefined,
      profession: user ? (user as any).profession ?? undefined : undefined,
      cachedAt:   Date.now(),
    };

    // Cache negative results for 30s — prevents repeated DB spam on reconnecting
    // clients while still allowing role upgrades to propagate quickly.
    // Cap the map so the connection history can't leak memory unbounded
    // (same oldest-entry eviction as the intent cache).
    if (this.profileCache.size >= 5000) {
      const oldest = this.profileCache.keys().next().value as string;
      this.profileCache.delete(oldest);
    }
    if (!user) {
      this.profileCache.set(uid, {
        ...profile,
        cachedAt: Date.now() - PROFILE_CACHE_TTL_MS + 30_000,
      });
    } else {
      this.profileCache.set(uid, profile);
    }

    return profile;
  }
}
