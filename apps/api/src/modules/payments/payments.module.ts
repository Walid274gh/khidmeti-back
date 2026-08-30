import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { Payment, PaymentSchema } from '../../schemas/payment.schema';
import { User, UserSchema } from '../../schemas/user.schema';
import { UsersModule } from '../users/users.module';
import { PaymentsService } from './payments.service';
import { PaymentsController } from './payments.controller';
import { SatimGateway } from './gateway/satim.gateway';

@Module({
  imports: [
    MongooseModule.forFeature([{ name: Payment.name, schema: PaymentSchema }]),
    MongooseModule.forFeature([{ name: User.name, schema: UserSchema }]),
    UsersModule,
  ],
  controllers: [PaymentsController],
  providers: [PaymentsService, SatimGateway],
  exports: [PaymentsService],
})
export class PaymentsModule {}
