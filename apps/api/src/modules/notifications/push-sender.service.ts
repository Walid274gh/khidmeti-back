// apps/api/src/modules/notifications/push-sender.service.ts
//
// The single place that turns a domain event into a user-facing notification:
//   1. Persists it to the Mongo `notifications` collection (in-app inbox).
//   2. Sends an FCM push to the recipient's registered device token.
//
// Both steps are best-effort and fully isolated — a failure in either NEVER
// propagates to the caller. Business mutations (bids, jobs) call `notify(...)`
// fire-and-forget alongside the WebSocket emit, so realtime + push + inbox all
// fire from the same point.
//
// The `type` + `data.requestId` fields mirror the Flutter client's
// `_routeForNotification()` contract so a tapped push lands on the right screen.

import { Injectable, Logger } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { admin } from '../../config/firebase.config';
import { User, UserDocument } from '../../schemas/user.schema';
import { NotificationsService } from './notifications.service';
import {
  buildNotification,
  NotificationParams,
  NotificationText,
} from './notification-messages';

export interface NotifyEvent {
  /** Routing key consumed by the app, e.g. 'bid_received' | 'bid_accepted'. */
  type: string;
  /** Params for the localized template, e.g. { workerName, price }. */
  params?: NotificationParams;
  /** Extra string data (FCM data values must be strings), e.g. { requestId }. */
  data?: Record<string, string>;
}

@Injectable()
export class PushSenderService {
  private readonly logger = new Logger(PushSenderService.name);

  constructor(
    @InjectModel(User.name)
    private readonly userModel: Model<UserDocument>,
    private readonly notifications: NotificationsService,
  ) {}

  /**
   * Persist + push to a single user, in that user's language. Never throws —
   * safe to call fire-and-forget (`void this.pushSender.notify(...)`).
   */
  async notify(userId: string, event: NotifyEvent): Promise<void> {
    // One lookup for the FCM token, the recipient's language and their mutes.
    let token: string | null | undefined;
    let lang = 'fr';
    let muted: string[] = [];
    try {
      const user = await this.userModel
        .findById(userId)
        .select('fcmToken language mutedNotifTypes')
        .lean()
        .exec();
      token = (user as { fcmToken?: string | null } | null)?.fcmToken;
      lang = (user as { language?: string } | null)?.language ?? 'fr';
      muted = (user as { mutedNotifTypes?: string[] } | null)?.mutedNotifTypes ?? [];
    } catch (err) {
      this.logger.warn(
        `notify: user lookup failed for ${userId}: ${(err as Error).message}`,
      );
    }

    const text = buildNotification(event.type, lang, event.params);

    // Mutes gate the PUSH only — the inbox keeps the full history, so a user
    // who mutes a type can still find the event on the notifications screen.
    await this.persist(userId, event.type, text, event.data);
    if (muted.includes(event.type)) return;
    await this.sendFcm(userId, token, event.type, text, event.data);
  }

  /**
   * Persist + push a caller-supplied (non-templated) message to one user. Used
   * by the admin broadcast feature where the title/body are authored in the
   * dashboard rather than derived from a NotificationText template. Never throws.
   */
  async notifyRaw(
    userId: string,
    payload: { title: string; body: string; type?: string; data?: Record<string, string> },
  ): Promise<void> {
    const type = payload.type ?? 'admin_broadcast';
    const text: NotificationText = { title: payload.title, body: payload.body };

    let token: string | null | undefined;
    let muted: string[] = [];
    try {
      const user = await this.userModel
        .findById(userId)
        .select('fcmToken mutedNotifTypes')
        .lean()
        .exec();
      token = (user as { fcmToken?: string | null } | null)?.fcmToken;
      muted = (user as { mutedNotifTypes?: string[] } | null)?.mutedNotifTypes ?? [];
    } catch (err) {
      this.logger.warn(
        `notifyRaw: user lookup failed for ${userId}: ${(err as Error).message}`,
      );
    }

    await this.persist(userId, type, text, payload.data);
    if (muted.includes(type)) return;
    await this.sendFcm(userId, token, type, text, payload.data);
  }

  // ── In-app inbox ─────────────────────────────────────────────────────────

  private async persist(
    userId: string,
    type: string,
    text: NotificationText,
    data?: Record<string, string>,
  ): Promise<void> {
    try {
      await this.notifications.create({
        userId,
        title: text.title,
        body: text.body,
        type,
        data: data ?? {},
      });
    } catch (err) {
      this.logger.warn(
        `notify: inbox persist failed for ${userId}: ${(err as Error).message}`,
      );
    }
  }

  // ── FCM push ─────────────────────────────────────────────────────────────

  /**
   * Android channel for a background-rendered notification. Channel ids and
   * importances are created by the Flutter client (NotificationService):
   * actionable marketplace events ride the high-importance channels so they
   * heads-up; everything else stays on the default-importance general channel.
   * Without an explicit id, FCM falls back to the manifest default
   * (general_channel) and bids/jobs never banner.
   */
  private static channelFor(type: string): string {
    if (type === 'chat_message') return 'messages_channel';
    if (
      type.startsWith('bid_') ||
      type.startsWith('job_') ||
      type.startsWith('request_') ||
      type.startsWith('rating_') ||
      type === 'new_request'
    ) {
      return 'service_requests_channel';
    }
    return 'general_channel';
  }

  private async sendFcm(
    userId: string,
    token: string | null | undefined,
    type: string,
    text: NotificationText,
    data?: Record<string, string>,
  ): Promise<void> {
    if (!token) return; // no device registered — inbox entry still exists

    // iOS badge = unread inbox count, computed AFTER persist() so the event
    // being pushed is included. A count failure just omits the badge — never
    // blocks the send. (The app clears the badge on open; the next push
    // re-syncs it to this server-side truth.)
    let badge: number | undefined;
    try {
      badge = await this.notifications.countUnread(userId);
    } catch {
      badge = undefined;
    }

    try {
      await admin.messaging().send({
        token,
        notification: { title: text.title, body: text.body },
        // Data values MUST be strings for FCM.
        data: { type, ...(data ?? {}) },
        android: {
          priority: 'high',
          notification: { channelId: PushSenderService.channelFor(type) },
        },
        apns: {
          headers: { 'apns-priority': '10' },
          // Without an explicit aps.sound, iOS renders the alert silently.
          payload: {
            aps: { sound: 'default', ...(badge === undefined ? {} : { badge }) },
          },
        },
      });
    } catch (err) {
      await this.handleSendError(userId, err);
    }
  }

  private async handleSendError(userId: string, err: unknown): Promise<void> {
    const e = err as { code?: string; message?: string };
    // A stale/unregistered token will never succeed — clear it so we stop
    // paying for failed sends on every future event.
    if (
      e.code === 'messaging/registration-token-not-registered' ||
      e.code === 'messaging/invalid-registration-token'
    ) {
      await this.userModel
        .updateOne({ _id: userId }, { fcmToken: null })
        .exec()
        .catch(() => undefined);
      this.logger.log(`notify: cleared stale FCM token for ${userId}`);
      return;
    }
    this.logger.warn(`notify: FCM send failed for ${userId}: ${e.message}`);
  }
}
