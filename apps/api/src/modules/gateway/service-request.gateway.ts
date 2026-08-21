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
import { PinGateService } from '../auth/pin-gate.service';

interface AuthenticatedSocket extends Socket {
  data: { uid: string };
}

@WebSocketGateway({
  namespace: '/requests',
  cors: { origin: '*', credentials: false },
  transports: ['websocket', 'polling'],
})
export class ServiceRequestGateway
  implements OnGatewayConnection, OnGatewayDisconnect
{
  @WebSocketServer() private readonly server!: Server;
  private readonly logger = new Logger(ServiceRequestGateway.name);

  constructor(private readonly pinGate: PinGateService) {}

  // ── Connection lifecycle ───────────────────────────────────────────────────

  async handleConnection(socket: AuthenticatedSocket): Promise<void> {
    try {
      const token =
        socket.handshake.auth?.['token'] as string | undefined ??
        (socket.handshake.headers['authorization'] as string | undefined)?.replace('Bearer ', '');

      if (!token) {
        // Expected noise, not an attack: app builds predating the
        // connect-after-auth change handshake tokenless before sign-in.
        this.logger.debug(`[WS requests] Rejected unauthenticated socket ${socket.id}`);
        socket.disconnect(true);
        return;
      }

      const decoded = await admin.auth().verifyIdToken(token);

      // Account-PIN device gate — same rule as FirebaseAuthGuard for HTTP.
      const deviceId = socket.handshake.auth?.['deviceId'] as string | undefined;
      if (!(await this.pinGate.isDeviceAllowed(decoded.uid, deviceId))) {
        this.logger.warn(`[WS requests] PIN_REQUIRED — rejected ${decoded.uid} (socket ${socket.id})`);
        socket.disconnect(true);
        return;
      }

      // Document-approval gate — un-approved accounts get no realtime either.
      if (!(await this.pinGate.isApproved(decoded.uid))) {
        this.logger.warn(`[WS requests] APPROVAL_PENDING — rejected ${decoded.uid} (socket ${socket.id})`);
        socket.disconnect(true);
        return;
      }

      socket.data.uid = decoded.uid;

      // Auto-join the user's personal notification room
      await socket.join(`user:${decoded.uid}`);

      this.logger.log(`[WS requests] ${decoded.uid} connected (socket ${socket.id})`);
    } catch (err) {
      this.logger.warn(`[WS requests] Auth failure on socket ${socket.id}: ${err}`);
      socket.disconnect(true);
    }
  }

  handleDisconnect(socket: AuthenticatedSocket): void {
    this.logger.log(`[WS requests] Socket ${socket.id} (uid=${socket.data?.uid ?? 'unknown'}) disconnected`);
  }

  // ── Room subscriptions ─────────────────────────────────────────────────────

  /**
   * Subscribe to real-time updates for a specific service request.
   * Room: `request:{requestId}`
   * Used by: client tracking screen, worker job detail screen.
   */
  @SubscribeMessage('subscribe:request')
  async handleSubscribeRequest(
    @ConnectedSocket() socket: AuthenticatedSocket,
    @MessageBody() payload: { requestId: string },
  ): Promise<void> {
    if (!payload?.requestId || typeof payload.requestId !== 'string') return;
    await socket.join(`request:${payload.requestId}`);
    this.logger.debug(`[WS requests] ${socket.data.uid} joined request:${payload.requestId}`);
  }

  /**
   * Unsubscribe from a specific request room.
   */
  @SubscribeMessage('unsubscribe:request')
  async handleUnsubscribeRequest(
    @ConnectedSocket() socket: AuthenticatedSocket,
    @MessageBody() payload: { requestId: string },
  ): Promise<void> {
    if (!payload?.requestId) return;
    await socket.leave(`request:${payload.requestId}`);
  }

  /**
   * Worker subscribes to available requests in their wilaya+service combination.
   * Room: `wilaya:{wilayaCode}:service:{serviceType}`
   * Used by: worker browse screen.
   */
  @SubscribeMessage('subscribe:available_requests')
  async handleSubscribeAvailableRequests(
    @ConnectedSocket() socket: AuthenticatedSocket,
    @MessageBody() payload: { wilayaCode: number; serviceType: string },
  ): Promise<void> {
    if (!payload?.wilayaCode || !payload?.serviceType) return;
    const room = `wilaya:${payload.wilayaCode}:service:${payload.serviceType}`;
    await socket.join(room);
    this.logger.debug(`[WS requests] ${socket.data.uid} joined ${room}`);
  }

  // ── Server-initiated broadcast helpers ────────────────────────────────────
  // Called by NestJS services after Mongo writes to push updates immediately.

  /**
   * Notify the client who owns a request that a new bid arrived.
   * Targets: `user:{userId}` room + `request:{requestId}` room.
   */
  emitBidReceived(userId: string, requestId: string, bid: Record<string, unknown>): void {
    const event = { requestId, bid, ts: Date.now() };
    this.server.to(`user:${userId}`).emit('request:bid_received', event);
    this.server.to(`request:${requestId}`).emit('request:bid_received', event);
  }

  /**
   * Notify all parties that a request's status has changed.
   * Targets: `request:{requestId}` room.
   */
  emitRequestUpdated(
    requestId: string,
    update: Record<string, unknown>,
  ): void {
    this.server.to(`request:${requestId}`).emit('request:updated', {
      requestId,
      ...update,
      ts: Date.now(),
    });
  }

  /**
   * Notify a worker that a client accepted their bid.
   * Targets: `user:{workerId}` room.
   */
  emitBidAccepted(workerId: string, requestId: string, bidId: string): void {
    this.server.to(`user:${workerId}`).emit('bid:accepted', {
      requestId,
      bidId,
      ts: Date.now(),
    });
  }

  /**
   * Notify workers in a wilaya that a new request is available for bidding.
   * Targets: `wilaya:{wilayaCode}:service:{serviceType}` room.
   */
  emitNewAvailableRequest(
    wilayaCode: number,
    serviceType: string,
    request: Record<string, unknown>,
  ): void {
    const room = `wilaya:${wilayaCode}:service:${serviceType}`;
    this.server.to(room).emit('request:created', { request, ts: Date.now() });
  }

  /**
   * Notify the worker and the client when a job is started.
   */
  emitJobStarted(requestId: string, workerId: string, clientUserId: string): void {
    const event = { requestId, ts: Date.now() };
    this.server.to(`request:${requestId}`).emit('request:started', event);
    this.server.to(`user:${workerId}`).emit('request:started', event);
    this.server.to(`user:${clientUserId}`).emit('request:started', event);
  }

  /**
   * Notify both parties when a job is completed.
   */
  emitJobCompleted(requestId: string, workerId: string, clientUserId: string): void {
    const event = { requestId, ts: Date.now() };
    this.server.to(`request:${requestId}`).emit('request:completed', event);
    this.server.to(`user:${workerId}`).emit('request:completed', event);
    this.server.to(`user:${clientUserId}`).emit('request:completed', event);
  }

  /**
   * Notify both parties when a request is cancelled.
   */
  emitRequestCancelled(requestId: string, workerId?: string, clientUserId?: string): void {
    const event = { requestId, ts: Date.now() };
    this.server.to(`request:${requestId}`).emit('request:cancelled', event);
    if (workerId) this.server.to(`user:${workerId}`).emit('request:cancelled', event);
    if (clientUserId) this.server.to(`user:${clientUserId}`).emit('request:cancelled', event);
  }
}
