import { IsString, MaxLength } from 'class-validator';

export class UpdateFcmTokenDto {
  /**
   * FCM registration token. An EMPTY string clears the stored token —
   * the sign-out / notifications-off / account-deletion flows all send ''
   * (rejecting it used to 400 and leave stale tokens receiving pushes).
   */
  @IsString()
  @MaxLength(512)
  fcmToken: string;
}
