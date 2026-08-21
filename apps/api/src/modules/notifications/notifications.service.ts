import { Injectable, Logger, NotFoundException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { v4 as uuidv4 } from 'uuid';
import { Notification, NotificationDocument } from '../../schemas/notification.schema';

export interface CreateNotificationInput {
  userId: string;
  title: string;
  body: string;
  type: string;
  data?: Record<string, unknown>;
}

@Injectable()
export class NotificationsService {
  private readonly logger = new Logger(NotificationsService.name);

  constructor(
    @InjectModel(Notification.name)
    private readonly notifModel: Model<NotificationDocument>,
  ) {}

  async create(input: CreateNotificationInput): Promise<NotificationDocument> {
    try {
      const notif = new this.notifModel({
        _id:       uuidv4(),
        userId:    input.userId,
        title:     input.title,
        body:      input.body,
        type:      input.type,
        data:      input.data ?? {},
        createdAt: new Date(),
        isRead:    false,
      });
      return await notif.save();
    } catch (err) {
      this.logger.error('NotificationsService.create failed', err);
      throw err;
    }
  }

  async findForUser(userId: string, limit = 50): Promise<NotificationDocument[]> {
    try {
      return await this.notifModel
        .find({ userId })
        .sort({ createdAt: -1 })
        .limit(Math.min(limit, 100))
        .exec();
    } catch (err) {
      this.logger.error(`NotificationsService.findForUser(${userId}) failed`, err);
      throw err;
    }
  }

  async markRead(id: string, userId: string): Promise<void> {
    try {
      const result = await this.notifModel
        .updateOne({ _id: id, userId }, { isRead: true })
        .exec();
      if (result.matchedCount === 0) {
        throw new NotFoundException(`Notification ${id} not found`);
      }
    } catch (err) {
      this.logger.error(`NotificationsService.markRead(${id}) failed`, err);
      throw err;
    }
  }

  async markAllRead(userId: string): Promise<void> {
    try {
      await this.notifModel.updateMany({ userId, isRead: false }, { isRead: true }).exec();
    } catch (err) {
      this.logger.error(`NotificationsService.markAllRead(${userId}) failed`, err);
      throw err;
    }
  }

  async countUnread(userId: string): Promise<number> {
    try {
      return await this.notifModel.countDocuments({ userId, isRead: false }).exec();
    } catch (err) {
      this.logger.error(`NotificationsService.countUnread(${userId}) failed`, err);
      throw err;
    }
  }
}
