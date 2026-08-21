import { Module } from '@nestjs/common';
import { AuthModule }  from '../auth/auth.module';
import { UsersModule } from '../users/users.module';
import { ServiceRequestsModule } from '../service-requests/service-requests.module';
import { SearchController } from './search.controller';
import { SemanticSearchService } from './semantic-search.service';

@Module({
  // EmbeddingsService / QdrantInitService come from the global QdrantModule.
  imports:     [AuthModule, UsersModule, ServiceRequestsModule],
  controllers: [SearchController],
  providers:   [SemanticSearchService],
})
export class SearchModule {}
