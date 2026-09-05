// ══════════════════════════════════════════════════════════════════════════════
// ProfessionsService
//
// RESPONSABILITÉS :
//   1. findAll(lang)       — liste paginée triée par sortOrder
//   2. getCategories(lang) — groupement par catégorie pour le picker Flutter
//   3. seed()              — population initiale idempotente (onModuleInit)
//
// COHÉRENCE IA :
//   Les `key` en base correspondent exactement aux valeurs de VALID_PROFESSIONS
//   dans intent-extractor.service.ts — les deux doivent évoluer ensemble.
// ══════════════════════════════════════════════════════════════════════════════

import {
  Injectable,
  Logger,
  OnModuleInit,
} from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model }       from 'mongoose';
import { Profession, ProfessionDocument } from '../../schemas/profession.schema';
import { ProfessionDto, ProfessionCategoryDto } from './dto/profession.dto';
import { INITIAL_PROFESSIONS }            from './seeders/professions.seeder';

type Lang = 'fr' | 'ar' | 'en';

const VALID_LANGS = new Set<Lang>(['fr', 'ar', 'en']);

@Injectable()
export class ProfessionsService implements OnModuleInit {
  private readonly logger = new Logger(ProfessionsService.name);

  constructor(
    @InjectModel(Profession.name)
    private readonly professionModel: Model<ProfessionDocument>,
  ) {}

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  async onModuleInit(): Promise<void> {
    await this.seed();
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Retourne toutes les professions actives dans la langue demandée,
   * triées par sortOrder.
   *
   * @param lang  Langue cible (default: 'fr'). Toute valeur invalide est ramenée à 'fr'.
   */
  async findAll(lang: Lang = 'fr'): Promise<ProfessionDto[]> {
    const resolvedLang = this.resolvelang(lang);

    const docs = await this.professionModel
      .find({ isActive: true })
      .sort({ sortOrder: 1 })
      .lean()
      .exec();

    return docs.map((d) => this.toDto(d as unknown as Profession, resolvedLang));
  }

  /**
   * Retourne les professions actives groupées par catégorie, dans la langue
   * demandée. Chaque catégorie est triée par sortOrder interne.
   *
   * @param lang  Langue cible.
   */
  async getCategories(lang: Lang = 'fr'): Promise<ProfessionCategoryDto[]> {
    const resolvedLang = this.resolvelang(lang);
    const professions  = await this.findAll(resolvedLang);

    // Groupement ordonné — préserve l'ordre d'insertion (sortOrder déjà appliqué)
    const categoryMap = new Map<string, ProfessionCategoryDto>();

    for (const p of professions) {
      if (!categoryMap.has(p.categoryKey)) {
        categoryMap.set(p.categoryKey, {
          key:         p.categoryKey,
          label:       p.categoryLabel,
          professions: [],
        });
      }
      categoryMap.get(p.categoryKey)!.professions.push(p);
    }

    return Array.from(categoryMap.values());
  }

  // ── Seeder ─────────────────────────────────────────────────────────────────

  /**
   * Seed idempotent : insère les professions initiales uniquement si la
   * collection est vide. N'écrase JAMAIS les données existantes.
   *
   * Stratégie "insert-if-absent" plutôt que "upsert all" pour préserver les
   * modifications admin (sortOrder, labels personnalisés, etc.) faites en prod.
   */
  private async seed(): Promise<void> {
    try {
      const count = await this.professionModel.estimatedDocumentCount().exec();
      if (count > 0) {
        this.logger.debug(`Professions already seeded (${count} documents) — skipping`);
        return;
      }

      await this.professionModel.insertMany(INITIAL_PROFESSIONS, { ordered: false });
      this.logger.log(`✅ Seeded ${INITIAL_PROFESSIONS.length} professions`);
    } catch (err) {
      // Non-fatal — l'app fonctionne même si le seed échoue (la collection
      // peut déjà exister avec un index unique qui provoque un duplicate error)
      this.logger.warn(`Profession seed partial/skipped: ${(err as Error).message}`);
    }
  }

  // ── Mappers ────────────────────────────────────────────────────────────────

  /**
   * Incrémente le score de demande d'un métier — appelé à chaque service
   * request créée (fire-and-forget : la création ne doit jamais échouer
   * à cause du compteur).
   */
  async bumpPopularity(key: string): Promise<void> {
    try {
      await this.professionModel
        .updateOne({ key }, { $inc: { popularity: 1 } })
        .exec();
    } catch (err) {
      this.logger.warn(`bumpPopularity(${key}) failed (non-fatal): ${(err as Error).message}`);
    }
  }

  private toDto(doc: Profession, lang: Lang): ProfessionDto {
    return {
      key:           doc.key,
      iconName:      doc.iconName,
      categoryKey:   doc.categoryKey,
      label:         doc.labels[lang],
      categoryLabel: doc.categoryLabels[lang],
      sortOrder:     doc.sortOrder,
      popularity:    doc.popularity ?? 0,
      imageUrl:      doc.imageUrl,
    };
  }

  private resolvelang(lang: string): Lang {
    return VALID_LANGS.has(lang as Lang) ? (lang as Lang) : 'fr';
  }
}
