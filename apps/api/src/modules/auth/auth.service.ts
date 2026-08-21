// ══════════════════════════════════════════════════════════════════════════════
// AuthService
//
// DESIGN NOTE — Pourquoi ne pas utiliser UsersService ici ?
//
//   UsersModule imports AuthModule (pour FirebaseAuthGuard).
//   Si AuthModule importait UsersModule, on créerait une dépendance circulaire.
//
//   Solution : AuthService injecte UserModel directement via DatabaseModule
//   (@Global()), qui exporte MongooseModule.forFeature([User]). AuthModule
//   n'a donc pas besoin d'importer UsersModule.
//
//   Cette classe reste volontairement fine (single-responsibility) :
//   seule la vérification d'existence d'un profil lui appartient.
// ══════════════════════════════════════════════════════════════════════════════

import { Injectable, Logger } from '@nestjs/common';
import { InjectModel }        from '@nestjs/mongoose';
import { Model }              from 'mongoose';
import { User, UserDocument } from '../../schemas/user.schema';
import { PinGateService }     from './pin-gate.service';

export interface UserCheckResult {
  /** true si aucun profil MongoDB n'existe encore pour cet uid Firebase. */
  isNewUser: boolean;

  /**
   * Rôle actuel du profil : 'client' | 'worker'.
   * null si isNewUser === true.
   */
  role: string | null;

  /** Le compte a un PIN configuré (opt-in anti-recyclage de SIM). */
  hasPin: boolean;

  /** Le device appelant doit passer POST /auth/verify-pin avant tout accès. */
  pinRequired: boolean;

  /**
   * Statut d'approbation des documents : '' (approuvé) | 'pending' | 'rejected'.
   * L'app parque l'utilisateur sur l'écran "en attente d'approbation" tant que
   * la valeur n'est pas ''. C'est la sonde que cet écran poll.
   */
  verificationStatus: string;

  /** Note de l'admin en cas de rejet (affichée à l'utilisateur). */
  verificationNote: string;
}

@Injectable()
export class AuthService {
  private readonly logger = new Logger(AuthService.name);

  constructor(
    @InjectModel(User.name)
    private readonly userModel: Model<UserDocument>,
    private readonly pinGate: PinGateService,
  ) {}

  /**
   * Vérifie si un profil existe dans MongoDB pour l'uid Firebase donné,
   * et si ce device doit présenter le PIN du compte avant de continuer.
   *
   * Appelé par AuthController immédiatement après signInWithCredential pour
   * décider si le client doit afficher le flow d'onboarding (nouveau),
   * l'écran PIN (device inconnu sur compte protégé), ou l'accueil.
   */
  async checkUser(uid: string, deviceId: string | undefined): Promise<UserCheckResult> {
    const doc = await this.userModel
      .findById(uid)
      .select('role verificationStatus verificationNote')
      .lean()
      .exec();

    const pin = await this.pinGate.status(uid, deviceId);

    const d = doc as unknown as {
      role?: string;
      verificationStatus?: string;
      verificationNote?: string;
    } | null;

    const result: UserCheckResult = {
      isNewUser: doc === null,
      role:      d?.role ?? null,
      verificationStatus: d?.verificationStatus ?? '',
      verificationNote:   d?.verificationNote ?? '',
      ...pin,
    };

    if (result.isNewUser) {
      this.logger.log(`Auth check: new user uid=${uid} — onboarding required`);
    }
    if (result.pinRequired) {
      this.logger.log(`Auth check: PIN required for uid=${uid} (unknown device)`);
    }

    return result;
  }
}
