import {
  Controller,
  Get,
  HttpCode,
  HttpStatus,
  Param,
  Patch,
  Query,
  UseGuards,
} from '@nestjs/common';
import { FirebaseAuthGuard } from '../../common/guards/firebase-auth.guard';
import { CurrentUser } from '../../common/decorators/current-user.decorator';
import { AuthUser } from '../../common/guards/firebase-auth.guard';
import { NotificationsService } from './notifications.service';
import { NotificationDocument } from '../../schemas/notification.schema';

@Controller('notifications')
@UseGuards(FirebaseAuthGuard)
export class NotificationsController {
  constructor(private readonly notificationsService: NotificationsService) {}

  /**
   * GET /notifications
   * Fetch the authenticated user's notifications.
   */
  @Get()
  async findForUser(
    @CurrentUser() user: AuthUser,
    @Query('limit') limitStr?: string,
  ): Promise<NotificationDocument[]> {
    const limit = (limitStr ? parseInt(limitStr, 10) : 50) || 50; // NaN/0 → 50
    return this.notificationsService.findForUser(user.uid, limit);
  }

  /**
   * GET /notifications/unread-count
   * Returns the number of unread notifications.
   */
  @Get('unread-count')
  async countUnread(@CurrentUser() user: AuthUser): Promise<{ count: number }> {
    const count = await this.notificationsService.countUnread(user.uid);
    return { count };
  }

  /**
   * PATCH /notifications/:id/read
   * Mark a single notification as read.
   */
  @Patch(':id/read')
  @HttpCode(HttpStatus.NO_CONTENT)
  async markRead(
    @Param('id') id: string,
    @CurrentUser() user: AuthUser,
  ): Promise<void> {
    return this.notificationsService.markRead(id, user.uid);
  }

  /**
   * PATCH /notifications/read-all
   * Mark all notifications as read for the current user.
   */
  @Patch('read-all')
  @HttpCode(HttpStatus.NO_CONTENT)
  async markAllRead(@CurrentUser() user: AuthUser): Promise<void> {
    return this.notificationsService.markAllRead(user.uid);
  }
}
