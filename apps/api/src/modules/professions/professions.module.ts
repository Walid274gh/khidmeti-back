import { Module } from '@nestjs/common';
import { MongooseModule }        from '@nestjs/mongoose';
import { Profession, ProfessionSchema } from '../../schemas/profession.schema';
import { ProfessionsService }    from './professions.service';
import { ProfessionsController } from './professions.controller';

@Module({
  imports: [
    MongooseModule.forFeature([
      { name: Profession.name, schema: ProfessionSchema },
    ]),
  ],
  controllers: [ProfessionsController],
  providers:   [ProfessionsService],
  exports:     [ProfessionsService],
})
export class ProfessionsModule {}
