import { IsOptional, Matches } from 'class-validator';

const PIN_RULE = /^\d{6}$/;
const PIN_MSG  = 'pin must be exactly 6 digits';

export class VerifyPinDto {
  @Matches(PIN_RULE, { message: PIN_MSG })
  pin: string;
}

export class SetPinDto {
  @Matches(PIN_RULE, { message: PIN_MSG })
  pin: string;

  /** Required when a PIN already exists (changing it). */
  @IsOptional()
  @Matches(PIN_RULE, { message: PIN_MSG })
  currentPin?: string;
}

export class RemovePinDto {
  @Matches(PIN_RULE, { message: PIN_MSG })
  currentPin: string;
}
