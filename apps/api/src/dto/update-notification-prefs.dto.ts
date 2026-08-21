import { ArrayMaxSize, IsArray, IsString, MaxLength } from 'class-validator';

export class UpdateNotificationPrefsDto {
  /**
   * Notification `type` keys to mute (push suppressed server-side, inbox
   * unaffected). The client sends the FULL muted list on every change —
   * last write wins, no merge semantics.
   */
  @IsArray()
  @ArrayMaxSize(32)
  @IsString({ each: true })
  @MaxLength(64, { each: true })
  mutedNotifTypes: string[];
}
