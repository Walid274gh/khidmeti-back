import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { ContactService } from './contact.service';
import { PushSenderService } from '../notifications/push-sender.service';
import { UsersService } from '../users/users.service';

/**
 * 48h rating nudge — setInterval, not a cron library.
 *
 * Why: the backend has zero scheduler infra (@nestjs/schedule is NOT
 * installed) and the sweep is one indexed Mongo query per interval —
 * adding a dependency for that is over-engineering. The runner is
 * single-flight (overlapping ticks skip) and each reminder is claimed
 * (reminderSent=true) BEFORE its push so a crash can only skip, never
 * double-send. FCM failures never roll the claim back — inbox is the
 * fallback, the client re-derives eligibility from rateableUntil anyway.
 */
@Injectable()
export class ContactReminderService {
  private readonly logger = new Logger(ContactReminderService.name);
  private running = false;

  constructor(
    private readonly contact: ContactService,
    private readonly pushSender: PushSenderService,
    private readonly usersService: UsersService,
    private readonly config: ConfigService,
  ) {
    const minutes = Number(this.config.get('RATING_REMINDER_INTERVAL_MIN') ?? 60);
    const ms = Math.max(5, minutes) * 60 * 1000;
    setInterval(() => void this.tick(), ms);
  }

  async tick(now = new Date()): Promise<void> {
    if (this.running) return;
    this.running = true;
    try {
      const due = await this.contact.dueReminders(now);
      for (const interaction of due) {
        // Claim first: a crash after this can only skip a nudge, never repeat it.
        await this.contact.markReminderSent(interaction._id);
        const worker = await this.usersService.findByIdOrNull(interaction.workerId);
        void this.pushSender.notify(interaction.clientId, {
          type: 'rating_reminder',
          params: { workerName: worker?.name },
          data: { contactId: interaction._id, requestId: interaction.requestId },
        });
      }
      if (due.length > 0) this.logger.log(`rating reminders sent: ${due.length}`);
    } catch (err) {
      this.logger.warn(`contact reminder tick failed (non-fatal): ${(err as Error).message}`);
    } finally {
      this.running = false;
    }
  }
}
