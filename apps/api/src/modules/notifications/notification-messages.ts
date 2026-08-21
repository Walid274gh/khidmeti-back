// apps/api/src/modules/notifications/notification-messages.ts
//
// Server-side localization for push/inbox notifications.
//
// WHY SERVER-SIDE: a background / killed app cannot run Dart to localize an FCM
// `notification` message — the OS renders whatever title/body the server sent.
// So the text must already be in the recipient's language. The user's language
// is stored on the User document (`language`, defaults to 'fr') and sent by the
// client on profile create/update.
//
// Keys MUST stay in sync with the Flutter client's `_routeForNotification`
// switch (main.dart) so a tapped push routes to the right screen.

export type NotificationLang = 'fr' | 'ar' | 'en';

export interface NotificationText {
  title: string;
  body: string;
}

export interface NotificationParams {
  workerName?: string;
  price?: number | string;
  /** Display name of the account that submitted verification documents. */
  name?: string;
  /** Admin's rejection note — authored text, passed through untranslated. */
  note?: string;
}

type Template = (p: NotificationParams) => NotificationText;

const TEMPLATES: Record<string, Record<NotificationLang, Template>> = {
  bid_received: {
    fr: (p) => ({
      title: 'Nouvelle offre reçue',
      body: `${p.workerName ?? 'Un prestataire'} a proposé ${p.price} DA`,
    }),
    ar: (p) => ({
      title: 'عرض جديد',
      body: `${p.workerName ?? 'مقدّم خدمة'} عرض ${p.price} دج`,
    }),
    en: (p) => ({
      title: 'New bid received',
      body: `${p.workerName ?? 'A provider'} offered ${p.price} DA`,
    }),
  },
  bid_accepted: {
    fr: () => ({
      title: 'Offre acceptée',
      body: 'Votre offre a été acceptée. Consultez les détails du travail.',
    }),
    ar: () => ({
      title: 'تم قبول عرضك',
      body: 'تم قبول عرضك. اطّلع على تفاصيل العمل.',
    }),
    en: () => ({
      title: 'Bid accepted',
      body: 'Your bid was accepted. Check the job details.',
    }),
  },
  bid_declined: {
    fr: () => ({
      title: 'Offre non retenue',
      body: 'Le client a choisi une autre offre pour cette demande.',
    }),
    ar: () => ({
      title: 'لم يتم اختيار عرضك',
      body: 'اختار العميل عرضاً آخر لهذا الطلب.',
    }),
    en: () => ({
      title: 'Bid not selected',
      body: 'The client chose another bid for this request.',
    }),
  },
  job_cancelled: {
    fr: () => ({
      title: 'Demande annulée',
      body: 'Le client a annulé la demande qui vous était attribuée.',
    }),
    ar: () => ({
      title: 'تم إلغاء الطلب',
      body: 'ألغى العميل الطلب الذي أُسند إليك.',
    }),
    en: () => ({
      title: 'Request cancelled',
      body: 'The client cancelled the request assigned to you.',
    }),
  },
  job_started: {
    fr: () => ({
      title: 'Travail commencé',
      body: 'Le prestataire a commencé votre demande.',
    }),
    ar: () => ({
      title: 'بدأ العمل',
      body: 'بدأ مقدّم الخدمة تنفيذ طلبك.',
    }),
    en: () => ({
      title: 'Work started',
      body: 'The provider has started your request.',
    }),
  },
  job_completed: {
    fr: () => ({
      title: 'Travail terminé',
      body: 'Votre demande est terminée. Notez le prestataire.',
    }),
    ar: () => ({
      title: 'اكتمل العمل',
      body: 'اكتملت خدمتك. قيّم مقدّم الخدمة.',
    }),
    en: () => ({
      title: 'Work completed',
      body: 'Your request is complete. Please rate the provider.',
    }),
  },
  job_declined: {
    fr: () => ({
      title: 'Prestataire désisté',
      body: 'Le prestataire s’est désisté. Votre demande est de nouveau ouverte aux offres.',
    }),
    ar: () => ({
      title: 'انسحب مقدّم الخدمة',
      body: 'انسحب مقدّم الخدمة. طلبك مفتوح للعروض من جديد.',
    }),
    en: () => ({
      title: 'Provider withdrew',
      body: 'The provider withdrew. Your request is open for bids again.',
    }),
  },
  // ── Document verification (worker/business approval flow) ───────────────────
  verification_submitted: {
    // Sent to ADMINS — new documents awaiting review.
    fr: (p) => ({
      title: 'Nouveaux documents à vérifier',
      body: `${p.name ?? 'Un compte'} a soumis des documents de vérification.`,
    }),
    ar: (p) => ({
      title: 'مستندات جديدة للمراجعة',
      body: `${p.name ?? 'حساب'} أرسل مستندات للتحقق.`,
    }),
    en: (p) => ({
      title: 'New documents to review',
      body: `${p.name ?? 'An account'} submitted verification documents.`,
    }),
  },
  verification_approved: {
    // Sent to the USER — account cleared, sign-in unblocked.
    fr: () => ({
      title: 'Compte approuvé',
      body: 'Vos documents ont été validés. Bienvenue sur Khidmeti !',
    }),
    ar: () => ({
      title: 'تم قبول حسابك',
      body: 'تم التحقق من مستنداتك. مرحباً بك في خدمتي!',
    }),
    en: () => ({
      title: 'Account approved',
      body: 'Your documents were verified. Welcome to Khidmeti!',
    }),
  },
  verification_rejected: {
    // Sent to the USER — includes the admin's note so they can fix + resubmit.
    fr: (p) => ({
      title: 'Documents refusés',
      body: p.note?.trim()
        ? `Motif : ${p.note.trim()} — corrigez puis soumettez à nouveau.`
        : 'Vos documents n’ont pas été validés. Corrigez puis soumettez à nouveau.',
    }),
    ar: (p) => ({
      title: 'تم رفض المستندات',
      body: p.note?.trim()
        ? `السبب: ${p.note.trim()} — صحّح ثم أعد الإرسال.`
        : 'لم يتم قبول مستنداتك. صحّحها ثم أعد الإرسال.',
    }),
    en: (p) => ({
      title: 'Documents rejected',
      body: p.note?.trim()
        ? `Reason: ${p.note.trim()} — fix and resubmit.`
        : 'Your documents were not approved. Fix and resubmit.',
    }),
  },
};

const FALLBACK: NotificationText = { title: 'Khidmeti', body: '' };

/**
 * Resolve a localized notification. Unknown `type` or `lang` degrade safely
 * (unknown lang → 'fr'; unknown type → neutral fallback).
 */
export function buildNotification(
  type: string,
  lang: string,
  params: NotificationParams = {},
): NotificationText {
  const byLang = TEMPLATES[type];
  if (!byLang) return FALLBACK;
  const normalized: NotificationLang =
    lang === 'ar' || lang === 'en' ? lang : 'fr';
  return byLang[normalized](params);
}
