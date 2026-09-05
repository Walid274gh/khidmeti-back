// apps/api/src/app.module.ts

import { Module } from '@nestjs/common';
import { APP_GUARD } from '@nestjs/core';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { MongooseModule }  from '@nestjs/mongoose';
import { ThrottlerModule, ThrottlerGuard } from '@nestjs/throttler';
import { DatabaseModule }           from './database/database.module';
import { QdrantModule }             from './qdrant/qdrant.module';
import { FirebaseConfigModule }     from './config/firebase.config';
import { AiModule }                 from './modules/ai/ai.module';
import { MediaModule }              from './modules/media/media.module';
import { UsersModule }              from './modules/users/users.module';
import { WorkersModule }            from './modules/workers/workers.module';
import { ServiceRequestsModule }    from './modules/service-requests/service-requests.module';
import { BidsModule }               from './modules/bids/bids.module';
import { ContactModule }            from './modules/contact/contact.module';
import { LocationModule }           from './modules/location/location.module';
import { NotificationsModule }      from './modules/notifications/notifications.module';
import { GatewayModule }            from './modules/gateway/gateway.module';
import { ProfessionsModule }        from './modules/professions/professions.module';
import { AdminModule }              from './modules/admin/admin.module';
import { SearchModule }             from './modules/search/search.module';
import { PaymentsModule }           from './modules/payments/payments.module';
import { HealthController }         from './health.controller';

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true, envFilePath: '../../.env' }),
    FirebaseConfigModule,

    MongooseModule.forRootAsync({
      imports:    [ConfigModule],
      inject:     [ConfigService],
      useFactory: (config: ConfigService) => ({
        uri:                      config.getOrThrow<string>('MONGODB_URI'),
        maxPoolSize:              10,
        serverSelectionTimeoutMS: 5000,
        socketTimeoutMS:          45000,
        // FIX: Suppress Mongoose 8.x autoIndex warnings on _id fields.
        // The 'users', 'worker_bids', etc. collections use string _id (Firebase UID).
        // Mongoose tries to add a sparse index on top of MongoDB's default _id index,
        // which logs "Warning: Can not overwrite the default `_id` index".
        // autoIndex: false in production means Mongoose won't attempt to sync indexes
        // on startup — run `db.collection.createIndexes()` in your migration instead.
        autoIndex: process.env['NODE_ENV'] !== 'production',
      }),
    }),

    ThrottlerModule.forRoot([
      { name: 'short',  ttl: 1_000,  limit: 20  },
      { name: 'medium', ttl: 10_000, limit: 100 },
      { name: 'long',   ttl: 60_000, limit: 300 },
    ]),

    DatabaseModule,
    QdrantModule,
    AiModule,
    MediaModule,
    UsersModule,
    WorkersModule,
    ServiceRequestsModule,
    BidsModule,
    ContactModule,
    LocationModule,
    NotificationsModule,
    GatewayModule,
    ProfessionsModule,
    AdminModule,
    SearchModule,
    PaymentsModule,
  ],
  controllers: [HealthController],
  // ThrottlerGuard must be registered globally or every @Throttle() decorator
  // (incl. the 10/min limit on /auth/check) is silently a no-op.
  // ponytail: in-memory store — fine for the single api container. If the api
  // is ever scaled to N replicas, limits multiply by N: switch to
  // @nest-lab/throttler-storage-redis (ioredis is already a dependency).
  providers: [{ provide: APP_GUARD, useClass: ThrottlerGuard }],
})
export class AppModule {}
