import { Module, Global } from '@nestjs/common';
import { QdrantInitService } from './qdrant-init.service';
import { EmbeddingsService } from './embeddings.service';
import { VectorIndexService } from './vector-index.service';

@Global()
@Module({
  providers: [QdrantInitService, EmbeddingsService, VectorIndexService],
  exports: [QdrantInitService, EmbeddingsService, VectorIndexService],
})
export class QdrantModule {}
