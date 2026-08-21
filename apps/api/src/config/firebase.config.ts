import { Module, OnModuleInit } from '@nestjs/common';
import * as admin from 'firebase-admin';

@Module({})
export class FirebaseConfigModule implements OnModuleInit {
  onModuleInit(): void {
    if (admin.apps.length > 0) return;

    const projectId = process.env['FIREBASE_PROJECT_ID'];
    const clientEmail = process.env['FIREBASE_CLIENT_EMAIL'];
    const privateKey = process.env['FIREBASE_PRIVATE_KEY']?.replace(/\\n/g, '\n');

    if (!projectId || !clientEmail || !privateKey) {
      throw new Error(
        'Firebase credentials missing. Set FIREBASE_PROJECT_ID, FIREBASE_CLIENT_EMAIL, FIREBASE_PRIVATE_KEY.',
      );
    }

    admin.initializeApp({
      credential: admin.credential.cert({ projectId, clientEmail, privateKey }),
    });

    console.log(`✅ Firebase Admin initialized — project: ${projectId}`);
  }
}

export { admin };
