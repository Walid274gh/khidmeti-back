import { IsBoolean } from 'class-validator';

export class UpdateAiConsentDto {
  /** Opt-in consent to log AI queries as training data (C1 flywheel). */
  @IsBoolean()
  aiDataConsent: boolean;
}
