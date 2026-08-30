# ══════════════════════════════════════════════════════════════════════════════
# apps/api/Dockerfile — v15
#
# v15 : ffmpeg SUPPRIMÉ (ne servait qu'au transcodage audio pour Gemma4 ;
# ai-stt décode tout format via PyAV). Image plus légère de ~30 MB.
# ══════════════════════════════════════════════════════════════════════════════

ARG NODE_VERSION=20.19-alpine3.21

# ══════════════════════════════════════════════════════════════════════════════
# Stage 0 : base — fondation commune à tous les stages
# ══════════════════════════════════════════════════════════════════════════════
FROM node:${NODE_VERSION} AS base

LABEL org.opencontainers.image.title="Khidmeti API"
LABEL org.opencontainers.image.description="NestJS backend — plateforme services à domicile Algérie"

WORKDIR /app

RUN chown node:node /app

USER node

# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 : deps — installation des dépendances
# ══════════════════════════════════════════════════════════════════════════════
FROM base AS deps

COPY --chown=node:node package.json ./

RUN npm install --legacy-peer-deps --ignore-scripts

# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 : builder — compilation TypeScript → dist/
# ══════════════════════════════════════════════════════════════════════════════
FROM deps AS builder

COPY --chown=node:node . .

RUN npm run build

# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 : development — hot-reload, source maps, ts-node
# ══════════════════════════════════════════════════════════════════════════════
FROM deps AS development

ENV NODE_ENV=development
ENV NODE_OPTIONS="--max-old-space-size=768"

COPY --chown=node:node . .

EXPOSE 3000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
  CMD wget -qO- http://localhost:3000/health || exit 1

CMD ["npm", "run", "start:dev"]

# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 : production — image minimale, sécurisée, sans devDeps
# ══════════════════════════════════════════════════════════════════════════════
FROM node:${NODE_VERSION} AS production

ARG NODE_VERSION=20.19-alpine3.21

WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3000
ENV NODE_OPTIONS="--max-old-space-size=768"

RUN apk add --no-cache wget \
 && apk upgrade --no-cache \
 && rm -rf /var/cache/apk/*

RUN chown node:node /app
USER node

COPY --chown=node:node package.json ./

RUN npm install --omit=dev --legacy-peer-deps --ignore-scripts \
 && npm cache clean --force

COPY --from=builder --chown=node:node /app/dist ./dist

EXPOSE 3000

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
  CMD wget -qO- http://localhost:${PORT:-3000}/health || exit 1

CMD ["node", "dist/main"]
