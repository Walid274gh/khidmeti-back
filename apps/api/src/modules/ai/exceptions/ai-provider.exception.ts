import { HttpException, HttpStatus } from '@nestjs/common';

export class AiRateLimitException extends HttpException {
  constructor() {
    super(
      { success: false, message: 'AI rate limit exceeded. Max 20 requests per hour.' },
      HttpStatus.TOO_MANY_REQUESTS,
    );
  }
}
