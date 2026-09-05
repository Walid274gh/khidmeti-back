// ══════════════════════════════════════════════════════════════════════════════
// Profession Schema
//
// DESIGN :
//   • Les professions sont gérées en BDD pour permettre l'ajout/désactivation
//     sans déploiement (via une interface admin future).
//   • Les `key` correspondent exactement aux valeurs utilisées par l'extracteur
//     d'intention IA (VALID_PROFESSIONS dans intent-extractor.service.ts) pour
//     assurer une cohérence totale entre le moteur IA et le picker Flutter.
//   • Les `labels` sont trilingues (fr/ar/en) pour le i18n Flutter.
//   • `iconName` est le nom d'une icône Material Icons (string) — mappé côté
//     Flutter, jamais côté serveur — ce qui permet des mises à jour d'icônes
//     sans modifier le backend.
//   • `sortOrder` contrôle l'ordre d'affichage dans le picker.
//   • `isActive: false` permet de masquer une profession sans la supprimer
//     (historique des demandes conservé).
// ══════════════════════════════════════════════════════════════════════════════

import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document } from 'mongoose';

export type ProfessionDocument = Profession & Document;

/** Clés de catégorie — stables, utilisées comme identifiants côté Flutter. */
export enum ProfessionCategory {
  Water     = 'water',
  Energy    = 'energy',
  Building  = 'building',
  Service   = 'service',
  Transport = 'transport',
}

@Schema({ collection: 'professions', timestamps: false, versionKey: false })
export class Profession {
  /**
   * Clé unique en snake_case.
   * DOIT correspondre exactement aux valeurs dans VALID_PROFESSIONS
   * de intent-extractor.service.ts.
   */
  @Prop({ required: true, unique: true, index: true })
  key: string;

  /**
   * Nom d'icône Material Icons — mappé côté Flutter uniquement.
   * Ex: 'water_drop_outlined', 'bolt_outlined', 'cleaning_services_outlined'
   */
  @Prop({ required: true })
  iconName: string;

  /** Catégorie pour le groupement dans le picker. */
  @Prop({ required: true, enum: Object.values(ProfessionCategory), index: true })
  categoryKey: string;

  /** Peut être désactivé par un admin sans supprimer les demandes existantes. */
  @Prop({ default: true, index: true })
  isActive: boolean;

  /** Ordre d'affichage dans le picker (croissant). */
  @Prop({ default: 0 })
  sortOrder: number;

  /**
   * Score de demande réel — +1 à chaque service request créée pour ce métier
   * (voir ProfessionsService.bumpPopularity). Servi tel quel dans
   * GET /professions ; Flutter classe les 4 cartes d'accueil par ce score
   * dès qu'au moins une valeur est > 0, sinon ordre curated de lancement.
   */
  @Prop({ default: 0, min: 0 })
  popularity: number;

  /**
   * URL d'image pour la carte de la profession sur l'écran d'accueil.
   * Gérée par l'admin ou le script seed-professions.
   * Non requise — le client Flutter retombe sur l'icône accent si absente.
   */
  @Prop({ type: String })
  imageUrl?: string;

  /** Labels localisés. */
  @Prop({
    required: true,
    type: {
      fr: { type: String, required: true },
      ar: { type: String, required: true },
      en: { type: String, required: true },
    },
    _id: false,
  })
  labels: {
    fr: string;
    ar: string;
    en: string;
  };

  /** Labels de catégorie localisés. */
  @Prop({
    required: true,
    type: {
      fr: { type: String, required: true },
      ar: { type: String, required: true },
      en: { type: String, required: true },
    },
    _id: false,
  })
  categoryLabels: {
    fr: string;
    ar: string;
    en: string;
  };
}

export const ProfessionSchema = SchemaFactory.createForClass(Profession);

ProfessionSchema.index({ isActive: 1, sortOrder: 1 });
ProfessionSchema.index({ categoryKey: 1, isActive: 1, sortOrder: 1 });
ProfessionSchema.index({ isActive: 1, popularity: -1 });
