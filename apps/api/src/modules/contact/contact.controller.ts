import {
  Body,
  Controller,
  Get,
  HttpCode,
  HttpStatus,
  Param,
  Patch,
  Post,
  Query,
  UseGuards,
} from '@nestjs/common';
import { FirebaseAuthGuard } from '../../common/guards/firebase-auth.guard';
import { CurrentUser } from '../../common/decorators/current-user.decorator';
import { AuthUser } from '../../common/guards/firebase-auth.guard';
import { ContactService } from './contact.service';
import { ResolveContactDto } from './dto/resolve-contact.dto';
import { ContactInteractionDocument } from '../../schemas/contact-interaction.schema';

@Controller()
@UseGuards(FirebaseAuthGuard)
export class ContactController {
  constructor(private readonly contact: ContactService) {}

  /**
   * POST /bids/:id/accept
   * Client accepts a quote → contact unlocked → interaction created.
   * Same path the app already calls — the URL does not change.
   */
  @Post('bids/:id/accept')
  @HttpCode(HttpStatus.OK)
  async acceptQuote(
    @Param('id') id: string,
    @CurrentUser() user: AuthUser,
  ): Promise<{ interactionId: string }> {
    return this.contact.acceptQuote(id, user.uid);
  }

  /**
   * GET /contacts?clientId=…
   * The client's contact history — drives "My contacts" + manual rating.
   */
  @Get('contacts')
  async findForClient(
    @Query('clientId') clientId: string,
    @CurrentUser() user: AuthUser,
  ): Promise<ContactInteractionDocument[]> {
    return this.contact.findForClient(clientId, user.uid);
  }

  /**
   * PATCH /contacts/:id
   * Resolve: { outcome: 'hired', stars, tags? } or { outcome: 'nohire' }.
   */
  @Patch('contacts/:id')
  @HttpCode(HttpStatus.NO_CONTENT)
  async resolve(
    @Param('id') id: string,
    @Body() dto: ResolveContactDto,
    @CurrentUser() user: AuthUser,
  ): Promise<void> {
    return this.contact.resolve(id, user.uid, dto);
  }
}
