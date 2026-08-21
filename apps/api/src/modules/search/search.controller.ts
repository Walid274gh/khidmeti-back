// apps/api/src/modules/search/search.controller.ts

import {
  BadRequestException,
  Controller,
  ForbiddenException,
  Get,
  Query,
  UseGuards,
} from '@nestjs/common';
import { FirebaseAuthGuard, AuthUser } from '../../common/guards/firebase-auth.guard';
import { CurrentUser } from '../../common/decorators/current-user.decorator';
import { SearchWorkersQueryDto } from '../../dto/search-workers.query.dto';
import { SearchRequestsQueryDto } from '../../dto/search-requests.query.dto';
import {
  SemanticSearchService,
  ScoredRequest,
  ScoredWorker,
} from './semantic-search.service';
import { UsersService } from '../users/users.service';
import { UserRole } from '../../schemas/user.schema';

@Controller('search')
@UseGuards(FirebaseAuthGuard)
export class SearchController {
  constructor(
    private readonly semanticSearch: SemanticSearchService,
    private readonly usersService:   UsersService,
  ) {}

  /**
   * GET /search/workers?q=…&wilayaCodes=31,46&limit=20
   *
   * Free-text semantic worker search (fr/ar/darija). Results use the same
   * document shape as GET /workers, ordered by relevance, each carrying an
   * extra `matchScore` (cosine similarity). Visibility rules are identical
   * to GET /workers. Returns 503 SEMANTIC_SEARCH_UNAVAILABLE when the
   * embedding backend is down — clients fall back to structured browse.
   */
  @Get('workers')
  async searchWorkers(
    @Query() dto: SearchWorkersQueryDto,
    @CurrentUser() user: AuthUser,
  ): Promise<ScoredWorker[]> {
    // Business-account view is server-enforced from the viewer's persisted
    // role — same rule as WorkersController.findMany.
    const b2bOnly =
      (await this.usersService.getRole(user.uid)) === UserRole.Business;

    return this.semanticSearch.searchWorkers(dto.q, {
      wilayaCodes: dto.wilayaCodes,
      limit:       dto.limit ?? 20,
      b2bOnly,
    });
  }

  /**
   * GET /search/requests?q=…&wilayaCodes=31,46&limit=20
   *
   * Reverse direction: a worker searching for open requests that match a
   * skill set. `q` is optional — when absent, the worker's own profile
   * (profession + bio, the exact text their vector was indexed from) becomes
   * the query: "requests that match me" with no typing. Results use the
   * GET /service-requests document shape in relevance order, each carrying
   * `matchScore`. Same 503 SEMANTIC_SEARCH_UNAVAILABLE contract as above.
   */
  @Get('requests')
  async searchRequests(
    @Query() dto: SearchRequestsQueryDto,
    @CurrentUser() user: AuthUser,
  ): Promise<ScoredRequest[]> {
    // Worker-only surface, enforced from the persisted role — same principle
    // as the b2bOnly derivation above, never from client claims. One lean
    // query both authorizes and fetches the profile-as-query text.
    const worker = await this.usersService.getWorkerForGateway(user.uid);
    if (!worker) throw new ForbiddenException('WORKER_ONLY');

    const query =
      dto.q?.trim() ||
      [worker.profession, worker.bio]
        .filter((s): s is string => !!s && s.trim().length > 0)
        .join('. ');
    if (query.length < 2) throw new BadRequestException('EMPTY_QUERY');

    return this.semanticSearch.searchRequests(query, {
      wilayaCodes: dto.wilayaCodes,
      limit:       dto.limit ?? 20,
    });
  }
}
