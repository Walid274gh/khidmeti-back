import { IsString, IsOptional, IsIn, IsNumber, IsBoolean } from 'class-validator';
import { SUBSCRIPTION_TIERS, SubscriptionTier } from '../../../schemas/user.schema';

export class CreatePaymentIntentDto {
  @IsIn([...SUBSCRIPTION_TIERS])
  tier!: SubscriptionTier;

  @IsOptional()
  @IsNumber()
  hoursPerDay?: number;

  @IsOptional()
  @IsNumber()
  bidsPerMonth?: number;

  @IsOptional()
  @IsBoolean()
  priority?: boolean;

  @IsOptional()
  @IsBoolean()
  b2b?: boolean;
}
