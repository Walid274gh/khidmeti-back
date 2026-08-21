// ── DTOs de réponse publique (pas d'exposition des _id MongoDB) ───────────────

export class ProfessionDto {
  /** Clé stable — utilisée dans les requêtes Flutter et le moteur IA. */
  key: string;

  /** Nom d'icône Material Icons pour l'affichage Flutter. */
  iconName: string;

  /** Clé de catégorie pour le groupement UI. */
  categoryKey: string;

  /** Label dans la langue demandée. */
  label: string;

  /** Label de catégorie dans la langue demandée. */
  categoryLabel: string;

  /** Ordre d'affichage. */
  sortOrder: number;

  /**
   * URL d'image Cloudinary (ou toute URL publique) pour la carte d'accueil.
   * Absent si non configurée — Flutter affiche l'icône accent en fallback.
   */
  imageUrl?: string;
}

export class ProfessionCategoryDto {
  /** Clé stable de la catégorie (ex: 'water', 'energy'). */
  key: string;

  /** Label localisé de la catégorie. */
  label: string;

  /** Professions actives dans cette catégorie, triées par sortOrder. */
  professions: ProfessionDto[];
}
