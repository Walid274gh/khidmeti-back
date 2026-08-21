import {
  createParamDecorator,
  ExecutionContext,
  UnauthorizedException,
} from '@nestjs/common';
import { Request } from 'express';
import { AuthUser } from '../guards/firebase-auth.guard';

export const CurrentUser = createParamDecorator(
  (_data: unknown, ctx: ExecutionContext): AuthUser => {
    const request = ctx.switchToHttp().getRequest<Request>();
    // A plain Error would surface as HTTP 500. This "should never happen" when
    // FirebaseAuthGuard is applied, but if a route forgets the guard we return
    // a correct 401 rather than an opaque server error.
    if (!request.user) {
      throw new UnauthorizedException('No authenticated user on request');
    }
    return request.user;
  },
);
