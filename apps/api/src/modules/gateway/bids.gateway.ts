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
  namespace: '/bids',
  cors: { origin: '*', credentials: false },
  transports: ['websocket', 'polling'],
})
export class BidsGateway implements OnGatewayConnection, OnGatewayDisconnect {
  @WebSocketServer() private readonly server!: Server;
  private readonly logger = new Logger(BidsGateway.name);

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
        this.logger.debug(`[WS bids] Rejected unauthenticated socket ${socket.id}`);
        socket.disconnect(true);
        return;
      }

      const decoded = await admin.auth().verifyIdToken(token);

      // Account-PIN device gate — same rule as FirebaseAuthGuard for HTTP.
      const deviceId = socket.handshake.auth?.['deviceId'] as string | undefined;
      if (!(await this.pinGate.isDeviceAllowed(decoded.uid, deviceId))) {
        this.logger.warn(`[WS bids] PIN_REQUIRED — rejected ${decoded.uid} (socket ${socket.id})`);
        socket.disconnect(true);
        return;
      }

      // Document-approval gate — un-approved accounts get no realtime either.
      if (!(await this.pinGate.isApproved(decoded.uid))) {
        this.logger.warn(`[WS bids] APPROVAL_PENDING — rejected ${decoded.uid} (socket ${socket.id})`);
        socket.disconnect(true);
        return;
      }

      socket.data.uid = decoded.uid;

      // Auto-join the user's personal notification room
      await socket.join(`user:${decoded.uid}`);

      this.logger.log(`[WS bids] ${decoded.uid} connected (socket ${socket.id})`);
    } catch (err) {
      this.logger.warn(`[WS bids] Auth failure on socket ${socket.id}: ${err}`);
      socket.disconnect(true);
    }
  }

  handleDisconnect(socket: AuthenticatedSocket): void {
    this.logger.log(`[WS bids] Socket ${socket.id} (uid=${socket.data?.uid ?? 'unknown'}) disconnected`);
  }

  // ── Room subscriptions ─────────────────────────────────────────────────────

  /**
   * Subscribe to the live bid list for a specific request.
   * Room: `request:{requestId}:bids`
   * Used by: client bids list screen.
   */
  @SubscribeMessage('subscribe:bids')
  async handleSubscribeBids(
    @ConnectedSocket() socket: AuthenticatedSocket,
    @MessageBody() payload: { requestId: string },
  ): Promise<void> {
    if (!payload?.requestId || typeof payload.requestId !== 'string') return;
    await socket.join(`request:${payload.requestId}:bids`);
    this.logger.debug(`[WS bids] ${socket.data.uid} joined request:${payload.requestId}:bids`);
  }

  /**
   * Unsubscribe from a bid room.
   */
  @SubscribeMessage('unsubscribe:bids')
  async handleUnsubscribeBids(
    @ConnectedSocket() socket: AuthenticatedSocket,
    @MessageBody() payload: { requestId: string },
  ): Promise<void> {
    if (!payload?.requestId) return;
    await socket.leave(`request:${payload.requestId}:bids`);
  }

  /**
   * Subscribe to personal bid status updates (worker watching their own bids).
   * Room: `worker:{workerId}:bids`
   * Used by: worker my-bids tab.
   */
  @SubscribeMessage('subscribe:worker_bids')
  async handleSubscribeWorkerBids(
    @ConnectedSocket() socket: AuthenticatedSocket,
  ): Promise<void> {
    await socket.join(`worker:${socket.data.uid}:bids`);
  }

  // ── Server-initiated broadcast helpers ────────────────────────────────────

  /**
   * A new bid has been submitted on a request.
   * Targets: `request:{requestId}:bids` room + `user:{requestOwnerId}`.
   */
  emitBidSubmitted(
    requestId: string,
    requestOwnerId: string,
    bid: Record<string, unknown>,
  ): void {
    const event = { requestId, bid, ts: Date.now() };
    this.server.to(`request:${requestId}:bids`).emit('bid:submitted', event);
    this.server.to(`user:${requestOwnerId}`).emit('bid:submitted', event);
  }

  /**
   * A bid has been accepted.
   * Targets: `request:{requestId}:bids` + `worker:{workerId}:bids` + `user:{workerId}`.
   */
  emitBidAccepted(
    requestId: string,
    bidId: string,
    workerId: string,
  ): void {
    const event = { requestId, bidId, ts: Date.now() };
    this.server.to(`request:${requestId}:bids`).emit('bid:accepted', event);
    this.server.to(`worker:${workerId}:bids`).emit('bid:accepted', event);
    this.server.to(`user:${workerId}`).emit('bid:accepted', event);
  }

  /**
   * A bid has been withdrawn by the worker.
   * Targets: `request:{requestId}:bids` + `worker:{workerId}:bids`.
   */
  emitBidWithdrawn(
    requestId: string,
    bidId: string,
    workerId: string,
  ): void {
    const event = { requestId, bidId, ts: Date.now() };
    this.server.to(`request:${requestId}:bids`).emit('bid:withdrawn', event);
    this.server.to(`worker:${workerId}:bids`).emit('bid:withdrawn', event);
  }

  /**
   * All other pending bids declined after one was accepted.
   * Targets: `request:{requestId}:bids`.
   */
  emitOtherBidsDeclined(requestId: string): void {
    this.server.to(`request:${requestId}:bids`).emit('bids:others_declined', {
      requestId,
      ts: Date.now(),
    });
  }

  /**
   * Notify a specific worker their bid was declined.
   * Targets: `user:{workerId}` + `worker:{workerId}:bids`.
   */
  emitBidDeclined(workerId: string, requestId: string, bidId: string): void {
    const event = { requestId, bidId, ts: Date.now() };
    this.server.to(`user:${workerId}`).emit('bid:declined', event);
    this.server.to(`worker:${workerId}:bids`).emit('bid:declined', event);
  }
}
