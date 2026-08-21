// apps/api/src/modules/gateway/gateway.module.ts
//
// FIX (migration collection unifiée) :
//   Remplacé l'import depuis '../../schemas/worker.schema' (fichier supprimé)
//   par '../../schemas/user.schema'. WorkerLocationGateway utilise déjà
//   UserDocument / UserRole en interne — seule la déclaration MongooseModule
//   était encore ancrée sur l'ancien schéma Worker.

import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { User, UserSchema }              from '../../schemas/user.schema';
import { AuthModule }                    from '../auth/auth.module';
import { WorkerLocationGateway }         from './worker-location.gateway';
import { ServiceRequestGateway }         from './service-request.gateway';
import { BidsGateway }                   from './bids.gateway';

@Module({
  imports: [
    MongooseModule.forFeature([{ name: User.name, schema: UserSchema }]),
    // PinGateService: gateways enforce the account-PIN device gate at handshake.
    AuthModule,
  ],
  providers: [WorkerLocationGateway, ServiceRequestGateway, BidsGateway],
  exports:   [WorkerLocationGateway, ServiceRequestGateway, BidsGateway],
})
export class GatewayModule {}
