// ══════════════════════════════════════════════════════════════════════════════
// ProfessionsController
//
// ENDPOINTS PUBLICS — aucune authentification requise.
//
// Justification : les professions sont des données de référence statiques
// (ou quasi-statiques). Les exposer publiquement permet :
//   • Au splash screen Flutter de charger la liste avant login.
//   • Aux workers de voir la liste dans l'écran d'inscription sans JWT.
//   • Aux CDN/proxies de cacher les réponses côté edge (Cache-Control: public).
//
// Cache-Control: public, max-age=86400 (24h)
//   Valeur délibérément longue — les professions changent rarement.
//   Le Flutter SDK met aussi en cache localement (Hive/SharedPrefs, TTL 24h).
//   En cas de changement urgent, incrémenter la version de l'API ou vider
//   le cache CDN manuellement.
// ══════════════════════════════════════════════════════════════════════════════

import {
  Controller,
  Get,
  Header,
  HttpCode,
  HttpStatus,
  Query,
} from '@nestjs/common';
import { ProfessionsService }            from './professions.service';
import { ProfessionDto, ProfessionCategoryDto } from './dto/profession.dto';

type Lang = 'fr' | 'ar' | 'en';

@Controller('professions')
export class ProfessionsController {
  constructor(private readonly professionsService: ProfessionsService) {}

  /**
   * GET /professions?lang=fr|ar|en
   *
   * Liste plate de toutes les professions actives, triées par sortOrder.
   * Utilisée par le picker Flutter (mode liste + recherche textuelle).
   * Réponse mise en cache 24h côté client et CDN.
   *
   * Query params :
   *   lang  — Langue cible. Default: 'fr'. Accepte: 'fr' | 'ar' | 'en'.
   *           Toute valeur inconnue est ramenée à 'fr'.
   */
  @Get()
  @HttpCode(HttpStatus.OK)
  @Header('Cache-Control', 'public, max-age=86400, stale-while-revalidate=3600')
  async findAll(
    @Query('lang') lang: Lang = 'fr',
  ): Promise<ProfessionDto[]> {
    return this.professionsService.findAll(lang);
  }

  /**
   * GET /professions/categories?lang=fr|ar|en
   *
   * Professions groupées par catégorie — structure optimisée pour le picker
   * Flutter en mode "liste groupée par catégorie" (25+ professions).
   * Réponse mise en cache 24h côté client et CDN.
   */
  @Get('categories')
  @HttpCode(HttpStatus.OK)
  @Header('Cache-Control', 'public, max-age=86400, stale-while-revalidate=3600')
  async getCategories(
    @Query('lang') lang: Lang = 'fr',
  ): Promise<ProfessionCategoryDto[]> {
    return this.professionsService.getCategories(lang);
  }
}
