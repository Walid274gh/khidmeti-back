import { NestFactory } from '@nestjs/core';
import { ValidationPipe } from '@nestjs/common';
import { SwaggerModule, DocumentBuilder } from '@nestjs/swagger';
import { AppModule } from './app.module';
import { HttpExceptionFilter } from './common/filters/http-exception.filter';
import { ResponseInterceptor } from './common/interceptors/response.interceptor';

async function bootstrap(): Promise<void> {
  // Fail fast on misconfigured deploys: these are read lazily at runtime, so
  // without this check a missing key boots green and 500s at first signup.
  const encKey = process.env['FIELD_ENC_KEY'];
  if (!encKey || encKey.length !== 64) {
    throw new Error('FIELD_ENC_KEY must be 64 hex chars. Generate: openssl rand -hex 32');
  }
  if (!process.env['FIELD_ENC_PEPPER']) {
    throw new Error('FIELD_ENC_PEPPER is required. Generate: openssl rand -hex 32');
  }
  for (const v of ['CLOUDINARY_CLOUD_NAME', 'CLOUDINARY_API_KEY', 'CLOUDINARY_API_SECRET']) {
    if (!process.env[v]) throw new Error(`${v} is required`);
  }

  const app = await NestFactory.create(AppModule);

  // Behind nginx: honor X-Forwarded-For so rate limiting keys on the real
  // client IP instead of nginx's (otherwise all users share one bucket).
  app.getHttpAdapter().getInstance().set('trust proxy', 1);

  // Global validation pipe
  app.useGlobalPipes(
    new ValidationPipe({
      whitelist: true,
      forbidNonWhitelisted: true,
      transform: true,
      transformOptions: { enableImplicitConversion: true },
    }),
  );

  // Global exception filter
  app.useGlobalFilters(new HttpExceptionFilter());

  // Global response interceptor
  app.useGlobalInterceptors(new ResponseInterceptor());

  // CORS
  const corsOrigins = process.env['CORS_ORIGINS']?.split(',') ?? ['*'];
  app.enableCors({
    origin: process.env['NODE_ENV'] === 'production' ? corsOrigins : true,
    credentials: true,
  });

  // Swagger (dev only)
  if (process.env['NODE_ENV'] !== 'production') {
    const config = new DocumentBuilder()
      .setTitle('Khidmeti API')
      .setDescription('Khidmeti home services platform API')
      .setVersion('1.0')
      .addBearerAuth()
      .build();
    const document = SwaggerModule.createDocument(app, config);
    SwaggerModule.setup('api/docs', app, document);
  }

  // Graceful shutdown: drain Mongo/Socket.IO on SIGTERM (docker stop/redeploy)
  // instead of killing in-flight writes mid-transition.
  app.enableShutdownHooks();

  const port = parseInt(process.env['PORT'] ?? '3000', 10);
  await app.listen(port);
  console.log(`🚀 Khidmeti API running on port ${port}`);
  console.log('🤖 AI: self-hosted (ai-nlu :8013, ai-stt :8014, ai-vision :8015)');
}

void bootstrap();
