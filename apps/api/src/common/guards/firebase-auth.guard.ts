import {
  CanActivate,
  ExecutionContext,
  ForbiddenException,
  Injectable,
  SetMetadata,
  UnauthorizedException,
} from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { Request } from 'express';
import * as admin from 'firebase-admin';
import { PinGateService } from '../../modules/auth/pin-gate.service';

export interface AuthUser {
  uid: string;
  email: string | undefined;
  name: string | undefined;
}

declare module 'express' {
  interface Request {
    user?: AuthUser;
  }
}

/**
 * Marks a route as exempt from the account-PIN device gate. Only for the PIN
 * endpoints themselves (/auth/verify-pin, /auth/check, /auth/request-pin-reset)
 * — everything else stays gated or the gate is not a gate.
 */
export const SKIP_PIN_GATE = 'skipPinGate';
export const SkipPinGate = () => SetMetadata(SKIP_PIN_GATE, true);

/**
 * Marks a route as exempt from the document-approval gate. Only for the routes
 * an un-approved account legitimately needs: /auth/check (the probe the
 * pending screen polls), POST /users (initial submission + resubmission after
 * rejection) and /media/upload/* (re-uploading documents). Everything else
 * stays blocked until an admin approves.
 */
export const SKIP_APPROVAL_GATE = 'skipApprovalGate';
export const SkipApprovalGate = () => SetMetadata(SKIP_APPROVAL_GATE, true);

@Injectable()
export class FirebaseAuthGuard implements CanActivate {
  constructor(
    private readonly reflector: Reflector,
    private readonly pinGate: PinGateService,
  ) {}

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const request = context.switchToHttp().getRequest<Request>();
    const authHeader = request.headers['authorization'];

    if (!authHeader?.startsWith('Bearer ')) {
      throw new UnauthorizedException('Missing or invalid Authorization header');
    }

    const token = authHeader.slice(7);

    let uid: string;
    try {
      const decoded = await admin.auth().verifyIdToken(token);
      uid = decoded.uid;
      request.user = {
        uid: decoded.uid,
        email: decoded.email,
        name: decoded.name as string | undefined,
      };
    } catch {
      throw new UnauthorizedException('Invalid or expired Firebase token');
    }

    // ── Account-PIN device gate (anti SIM-recycling) ─────────────────────────
    // A recycled SIM passes phone-OTP with the original owner's uid; if that
    // account opted into a PIN, only devices that presented the PIN may pass.
    const skip = this.reflector.getAllAndOverride<boolean>(SKIP_PIN_GATE, [
      context.getHandler(),
      context.getClass(),
    ]);
    if (!skip) {
      const deviceId = request.headers['x-device-id'] as string | undefined;
      if (!(await this.pinGate.isDeviceAllowed(uid, deviceId))) {
        // Distinct code — the app catches it and shows the PIN screen.
        throw new ForbiddenException('PIN_REQUIRED');
      }
    }

    // ── Document-approval gate (worker optional, business mandatory) ─────────
    // An account that submitted verification documents is blocked everywhere
    // until an admin approves — same mechanism as the PIN gate. The app catches
    // the distinct code and parks the user on the pending-approval screen.
    const skipApproval = this.reflector.getAllAndOverride<boolean>(SKIP_APPROVAL_GATE, [
      context.getHandler(),
      context.getClass(),
    ]);
    if (!skipApproval && !(await this.pinGate.isApproved(uid))) {
      throw new ForbiddenException('APPROVAL_PENDING');
    }

    return true;
  }
}
