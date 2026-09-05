// ══════════════════════════════════════════════════════════════════════════════
// DatabaseModule — global Mongoose model registrations
//
// The Worker model has been removed. All user documents (clients AND workers)
// now live in the 'users' collection discriminated by the `role` field.
//
// Profession model added: liste de professions gérée en BDD pour permettre
// l'ajout/désactivation sans déploiement.
// ══════════════════════════════════════════════════════════════════════════════

import { Module, Global } from '@nestjs/common';
import { MongooseModule }  from '@nestjs/mongoose';
import { User, UserSchema }                     from '../schemas/user.schema';
import { ServiceRequest, ServiceRequestSchema } from '../schemas/service-request.schema';
import { WorkerBid, WorkerBidSchema }           from '../schemas/worker-bid.schema';
import { ContactInteraction, ContactInteractionSchema } from '../schemas/contact-interaction.schema';
import { Notification, NotificationSchema }     from '../schemas/notification.schema';
import { GeographicCell, GeographicCellSchema } from '../schemas/geographic-cell.schema';
import { Profession, ProfessionSchema }         from '../schemas/profession.schema';
import { AiQueryLog, AiQueryLogSchema }         from '../schemas/ai-query-log.schema';

const MODELS = MongooseModule.forFeature([
  { name: User.name,           schema: UserSchema           },
  { name: ServiceRequest.name, schema: ServiceRequestSchema },
  { name: WorkerBid.name,      schema: WorkerBidSchema      },
  { name: ContactInteraction.name, schema: ContactInteractionSchema },
  { name: Notification.name,   schema: NotificationSchema   },
  { name: GeographicCell.name, schema: GeographicCellSchema },
  { name: Profession.name,     schema: ProfessionSchema     },
  { name: AiQueryLog.name,     schema: AiQueryLogSchema     },
]);

@Global()
@Module({
  imports: [MODELS],
  exports: [MODELS],
})
export class DatabaseModule {}
