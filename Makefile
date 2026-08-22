## ══════════════════════════════════════════════════════════════════════════════
## KHIDMETI BACKEND — Makefile v15
##
## v15 : Gemma4 (llama.cpp ~5 GB) SUPPRIMÉ — IA 100% maison :
##   ai-nlu :8013 (dziriBERT), ai-stt :8014 (faster-whisper),
##   ai-vision :8015 (SigLIP2), ai-embed :8012 (nomic-embed)
##   Poids auto-téléchargés au premier boot dans docker/models/{nlu,stt,vision,embed}
##
## Setup initial :
##   make start              → Lance tout (premier boot : ~1 GB de poids à télécharger)
## ══════════════════════════════════════════════════════════════════════════════

SHELL := /bin/bash

# ── Détection OS / Arch ───────────────────────────────────────────────────────
OS   := $(shell uname -s 2>/dev/null || echo Windows_NT)
ARCH := $(shell uname -m 2>/dev/null || echo unknown)

ifeq ($(OS),Darwin)
  OPEN_CMD := open
else ifeq ($(OS),Windows_NT)
  OPEN_CMD := start
else
  OPEN_CMD := $(shell command -v xdg-open 2>/dev/null || echo "echo ")
endif

SED_I := $(shell \
  if sed --version 2>/dev/null | grep -q GNU; then echo "sed -i"; \
  else echo "sed -i ''"; fi)

LOCAL_IP := $(shell \
  ip route get 1 2>/dev/null | awk '{print $$7; exit}' || \
  ifconfig 2>/dev/null | awk '/inet /{print $$2}' | grep -v 127.0.0.1 | head -1 || \
  hostname -I 2>/dev/null | awk '{print $$1}' || \
  echo "127.0.0.1")

HOST     := $(shell hostname)
DATETIME := $(shell date +%Y%m%d-%H%M%S)
ARGS     ?=


.DEFAULT_GOAL := help

.PHONY: help start stop restart models \
        _ensure-env _ensure-dirs _ensure-models _ensure-docker \
        build rebuild logs logs-api logs-mongo logs-redis \
        logs-qdrant logs-minio logs-nginx logs-mongo-ui \
        logs-ai-nlu logs-ai-stt logs-ai-vision \
        health status ai-status dns \
        minio-buckets minio-console minio-list \
        test-api test-ai test-ai-audio test-ai-image \
        firewall backup restore \
        shell-api shell-mongo shell-redis shell-minio shell-qdrant \
        mongo-stats redis-info redis-flush clean-logs clean \
        prod-start prod-stop prod-update prod-logs check encrypt-pii \
        hf-deploy hf-status \
        scripts-promote-admin \
        tunnel-quick tunnel-stop tunnel-status \
        flutter-run ngrok ngrok-reset \
        scripts scripts-migrations scripts-seeds

help:
	@echo ""
	@echo "══════════════════════════════════════════════════════"
	@echo "  KHIDMETI v15 — IA auto-hébergée (nlu / stt / vision)"
	@echo "  OS : $(OS) | IP : $(LOCAL_IP)"
	@echo "══════════════════════════════════════════════════════"
	@echo ""
	@echo "  ⚠️  SETUP INITIAL :"
	@echo "  make start             ← Lance TOUT (1er boot : poids IA ~1 GB auto-téléchargés)"
	@echo "                           bypass menu : MODE=cloud|local make start"
	@echo "                           sans tunnel : NGROK=0 make start"
	@echo ""
	@echo "  [Quotidien]"
	@echo "  make stop              Arrêter"
	@echo "  make restart           Redémarrer"
	@echo "  make health            État des services"
	@echo "  make ai-status         État IA + modèles sur disque"
	@echo "  make logs              Tous les logs"
	@echo "  make dns               URLs + Flutter config"
	@echo ""
	@echo "  [Logs]"
	@echo "  make logs-ai-nlu / logs-ai-stt / logs-ai-vision"
	@echo ""
	@echo "  [Tests IA]"
	@echo "  make test-ai           Darija → JSON (ai-nlu direct)"
	@echo "  make test-ai-audio     Health ai-stt"
	@echo "  make test-ai-image     Health ai-vision"
	@echo ""
	@echo "  [Tunnel]"
	@echo "  make ngrok             Tunnel permanent"
	@echo ""
	@echo "  [Production]"
	@echo "  make check             Compile + tests jest (dans Docker)"
	@echo "  make prod-start        Compose prod (image production, nginx:80 seul)"
	@echo "  make encrypt-pii       Migration PII one-shot (après 1er prod-start)"
	@echo "  make hf-deploy         Déploie api+ai-embed sur HuggingFace Spaces"
	@echo "  make hf-status         État des Spaces + /health"
	@echo "  make scripts-promote-admin ARGS=\"--email x@y.com\"  Créer/promouvoir un admin"
	@echo "  make prod-update       Rebuild + redéploie l'API seule"
	@echo "  make prod-stop / prod-logs"
	@echo ""
	@echo "  [Scripts]"
	@echo "  make scripts           Migrations + Seeds"
	@echo "  make backup            Sauvegarder MongoDB"
	@echo ""
	@echo "  [Nettoyage]"
	@echo "  make clean             Volumes Docker (modèles intacts)"
	@echo ""

# ── _ensure-docker : échec précoce et clair si Docker manque ──────────────────
_ensure-docker:
	@command -v docker >/dev/null 2>&1 \
	  || { echo "  ❌ Docker absent → https://docs.docker.com/engine/install/"; exit 1; }
	@docker info >/dev/null 2>&1 \
	  || { echo "  ❌ Docker installé mais démon arrêté (ou permissions) — démarrez-le"; exit 1; }

# ── _ensure-dirs ──────────────────────────────────────────────────────────────
_ensure-dirs:
	@mkdir -p logs backups/mongodb docker/flywheel
	@chmod 777 docker/flywheel 2>/dev/null \
	  || echo "  ⚠️  docker/flywheel non inscriptible (api tourne en USER node) — lancez : sudo chown -R 1000:1000 docker/flywheel"

# ── _ensure-env ───────────────────────────────────────────────────────────────
_ensure-env:
	@if [ ! -f .env ]; then \
	  if [ -f .env.example ]; then \
	    cp .env.example .env; \
	    echo "  ⚠️  .env créé depuis .env.example → remplissez FIREBASE_* et HF_TOKEN dans .env"; \
	  else \
	    echo "  ⚠️  .env absent — créez-le manuellement"; \
	  fi; \
	fi

# ── _ensure-models ─────────────────────────────────────────────────────────────
# v15 : plus rien à pré-télécharger — chaque conteneur IA tire ses poids au
# premier boot (nlu ~130 MB HF privé, stt ~484 MB, vision ~470 MB, embed ~84 MB).
_ensure-models: _ensure-dirs
	@echo "  ℹ️  Poids IA : auto-téléchargés au premier boot (docker/models/…)"

models: _ensure-models

## ══════════════════════════════════════════════════════════════════════════════
## START
## ══════════════════════════════════════════════════════════════════════════════

## Menu cloud/local. Bypass non-interactif : MODE=cloud|local make start
## .env est LE fichier unique (infra cloud : Atlas/Upstash/Qdrant Cloud) — éditez-le directement.
start: _ensure-docker _ensure-dirs _ensure-models
	@echo "══════════════════════════════════════════════════════"
	@echo "  Démarrage Khidmeti v15 — IA auto-hébergée"
	@echo "══════════════════════════════════════════════════════"
	@echo ""
	@docker compose up -d --remove-orphans
	@echo ""
	@echo "  ⏳ Attente services IA (1er boot = téléchargement des poids, patiente)..."
	@READY=0; \
	for i in $$(seq 1 60); do \
	  if curl -sf http://localhost:8013/health >/dev/null 2>&1 \
	     && curl -sf http://localhost:8014/health >/dev/null 2>&1 \
	     && curl -sf http://localhost:8015/health >/dev/null 2>&1; then \
	    READY=1; break; fi; \
	  printf "."; sleep 5; \
	done; echo ""; \
	if [ "$$READY" -eq 1 ]; then \
	  echo "  ✅ ai-nlu + ai-stt + ai-vision prêts"; \
	else \
	  echo "  ⚠️  IA non prête → make logs-ai-nlu / logs-ai-stt / logs-ai-vision"; \
	fi
	@echo ""
	@$(MAKE) --no-print-directory health
	@echo ""
	@$(MAKE) --no-print-directory dns
	@if [ "$(NGROK)" != "0" ]; then \
	  echo ""; $(MAKE) --no-print-directory ngrok; \
	else \
	  echo "  (tunnel ngrok ignoré — NGROK=0)"; \
	fi

stop:
	@docker compose down --remove-orphans
	@echo ""
	@echo "  ✅ Services arrêtés. Modèles dans docker/models/ — intacts."
	@echo ""

restart: stop
	@sleep 2
	@$(MAKE) start

build:
	@docker compose build --no-cache api
	@echo "✅ Build terminé."

rebuild: build
	@$(MAKE) start

## ══════════════════════════════════════════════════════════════════════════════
## LOGS
## ══════════════════════════════════════════════════════════════════════════════

logs:
	@docker compose logs --tail=100 -f

logs-api:
	@docker compose logs -f api

logs-mongo:
	@docker compose logs -f mongo

logs-mongo-ui:
	@docker compose logs -f mongo-express

logs-redis:
	@docker compose logs -f redis

logs-qdrant:
	@docker compose logs -f qdrant

logs-minio:
	@docker compose logs -f minio

logs-nginx:
	@docker compose logs -f nginx

logs-ai-nlu:
	@docker compose logs -f ai-nlu

logs-ai-stt:
	@docker compose logs -f ai-stt

logs-ai-vision:
	@docker compose logs -f ai-vision

## ══════════════════════════════════════════════════════════════════════════════
## DIAGNOSTIC
## ══════════════════════════════════════════════════════════════════════════════

health:
	@echo ""
	@echo "══════════════════════════════════════════════════════"
	@echo "  État des services Khidmeti v14.5"
	@echo "══════════════════════════════════════════════════════"
	@echo ""
	@echo "  ── Local (Docker) ──────────────────────────────────"
	@_c() { \
	  code=$$(curl -s -o /dev/null -w "%{http_code}" -m 5 "$$2" 2>/dev/null); \
	  [ "$$code" = "200" ] && echo "  ✅ $$1" || echo "  ❌ $$1 (HTTP $$code → make logs-$$3)"; \
	}; \
	_c "NestJS    :3000" "http://localhost:3000/health" "api"; \
	_c "nginx     :80  " "http://localhost/health"      "nginx"; \
	_c "ai-nlu    :8013" "http://localhost:8013/health" "ai-nlu"; \
	_c "ai-stt    :8014" "http://localhost:8014/health" "ai-stt"; \
	_c "ai-vision :8015" "http://localhost:8015/health" "ai-vision"
	@echo ""
	@echo "  ── Cloud (managed) ─────────────────────────────────"
	@QURL=$$(grep '^QDRANT_URL=' .env 2>/dev/null | cut -d= -f2- | tr -d ' "'); \
	  QKEY=$$(grep '^QDRANT_API_KEY=' .env 2>/dev/null | cut -d= -f2- | tr -d ' "'); \
	  if [ -n "$$QURL" ]; then \
	    code=$$(curl -s -o /dev/null -w "%{http_code}" -m 8 -H "api-key: $$QKEY" "$$QURL/healthz" 2>/dev/null); \
	    [ "$$code" = "200" ] && echo "  ✅ Qdrant Cloud     (healthz 200)" \
	                        || echo "  ❌ Qdrant Cloud     (HTTP $$code — vérifier QDRANT_URL / QDRANT_API_KEY)"; \
	  else echo "  ⚠️  Qdrant Cloud     (QDRANT_URL absent du .env)"; fi
	@CN=$$(grep '^CLOUDINARY_CLOUD_NAME=' .env 2>/dev/null | cut -d= -f2- | tr -d ' "'); \
	  if [ -n "$$CN" ]; then \
	    code=$$(curl -s -o /dev/null -w "%{http_code}" -m 8 "https://res.cloudinary.com/$$CN/image/upload/sample.jpg" 2>/dev/null); \
	    [ "$$code" = "200" ] && echo "  ✅ Cloudinary       (CDN 200)" \
	                        || echo "  ❌ Cloudinary       (HTTP $$code — vérifier CLOUDINARY_CLOUD_NAME)"; \
	  else echo "  ⚠️  Cloudinary       (CLOUDINARY_CLOUD_NAME absent du .env)"; fi
	@MURI=$$(grep '^MONGODB_URI=' .env 2>/dev/null | cut -d= -f2- | tr -d ' "'); \
	  if command -v mongosh >/dev/null 2>&1 && [ -n "$$MURI" ]; then \
	    mongosh "$$MURI" --quiet --eval "db.adminCommand('ping').ok" >/dev/null 2>&1 \
	      && echo "  ✅ MongoDB Atlas    (ping ok)" \
	      || echo "  ❌ MongoDB Atlas    (ping échoué — vérifier MONGODB_URI / IP allowlist)"; \
	  else echo "  ☁️  MongoDB Atlas    (managed — état réel via NestJS /health/ready)"; fi
	@RURL=$$(grep '^REDIS_URL=' .env 2>/dev/null | cut -d= -f2- | tr -d ' "'); \
	  if command -v redis-cli >/dev/null 2>&1 && [ -n "$$RURL" ]; then \
	    redis-cli -u "$$RURL" ping 2>/dev/null | grep -qi PONG \
	      && echo "  ✅ Upstash Redis    (PONG)" \
	      || echo "  ❌ Upstash Redis    (pas de PONG — vérifier REDIS_URL)"; \
	  else echo "  ☁️  Upstash Redis    (managed — état réel via NestJS /health/ready)"; fi
	@echo ""

ai-status:
	@echo ""
	@echo "══════════════════════════════════════════════════════"
	@echo "  Statut IA v15 — modèles maison"
	@echo "══════════════════════════════════════════════════════"
	@echo ""
	@_c() { \
	  code=$$(curl -s -o /dev/null -w "%{http_code}" -m 5 "$$2" 2>/dev/null); \
	  [ "$$code" = "200" ] && echo "  ✅ $$1" || echo "  ❌ $$1 (→ make logs-$$3)"; \
	}; \
	_c "ai-nlu    :8013 (dziriBERT int8)" "http://localhost:8013/health" "ai-nlu"; \
	_c "ai-stt    :8014 (faster-whisper)" "http://localhost:8014/health" "ai-stt"; \
	_c "ai-vision :8015 (SigLIP2 fp32)"   "http://localhost:8015/health" "ai-vision"
	@echo ""
	@echo "  Poids sur disque :"
	@for d in nlu stt vision embed; do \
	  if [ -d "docker/models/$$d" ] && [ -n "$$(ls -A docker/models/$$d 2>/dev/null)" ]; then \
	    echo "  ✅ $$d/ ($$(du -sh docker/models/$$d 2>/dev/null | cut -f1))"; \
	  else \
	    echo "  ⬇️  $$d/ absent — téléchargé au premier boot"; \
	  fi; done
	@echo ""
	@free -h 2>/dev/null | awk '/^Mem:/{print "  RAM — Total: " $$2 "  Dispo: " $$7}' \
	  || echo "  (info RAM non disponible)"
	@echo ""

status:
	@echo ""
	@docker ps -a --filter "name=khidmeti" \
	  --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo ""
	@docker stats --no-stream \
	  --format "  {{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}" 2>/dev/null \
	  | grep khidmeti | sort || true
	@echo ""

dns:
	@echo ""
	@echo "══════════════════════════════════════════════════════"
	@echo "  URLs  [$(HOST)]"
	@echo "══════════════════════════════════════════════════════"
	@echo "  API      : http://$(HOST):3000"
	@echo "  nginx    : http://$(HOST):80"
	@echo "  Swagger  : http://$(HOST):3000/api/docs"
	@echo ""
	@echo "  ── Cloud consoles ───────────────────────────────────"
	@QURL=$$(grep '^QDRANT_URL=' .env 2>/dev/null | cut -d= -f2- | tr -d ' "'); \
	  CN=$$(grep '^CLOUDINARY_CLOUD_NAME=' .env 2>/dev/null | cut -d= -f2- | tr -d ' "'); \
	  [ -n "$$QURL" ] && echo "  Qdrant     : $$QURL/dashboard"; \
	  echo "  MongoDB    : https://cloud.mongodb.com  (Atlas)"; \
	  [ -n "$$CN" ] && echo "  Cloudinary : https://console.cloudinary.com  (cloud: $$CN)"; \
	  echo "  Upstash    : https://console.upstash.com  (Redis)"
	@echo ""
	@echo "  ── Flutter ──────────────────────────────────────────"
	@echo "  flutter run --dart-define=API_BASE_URL=http://$(LOCAL_IP):80"
	@NGROK_DOMAIN=$$(grep '^NGROK_DOMAIN=' .env 2>/dev/null | cut -d= -f2- | tr -d ' "'); \
	if [ -n "$$NGROK_DOMAIN" ]; then \
	  echo ""; \
	  echo "  ngrok   : https://$$NGROK_DOMAIN"; \
	  echo "  flutter run --dart-define=API_BASE_URL=https://$$NGROK_DOMAIN"; fi
	@echo ""

## ══════════════════════════════════════════════════════════════════════════════
## TESTS IA
## ══════════════════════════════════════════════════════════════════════════════

test-ai:
	@echo ""
	@echo "  Test ai-nlu — Darija → intent+profession..."
	@curl -s -X POST http://localhost:8013/classify \
	  -H "Content-Type: application/json" \
	  -d '{"text":"راني نحوس على بلومبي في سطيف"}' \
	  | python3 -m json.tool 2>/dev/null \
	  || echo "  ❌ ai-nlu non disponible → make logs-ai-nlu"
	@echo ""

test-ai-audio:
	@echo ""
	@code=$$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8014/health 2>/dev/null); \
	[ "$$code" = "200" ] \
	  && echo "  ✅ ai-stt:8014 opérationnel (faster-whisper small int8)" \
	  || echo "  ❌ ai-stt:8014 non disponible → make logs-ai-stt"
	@echo "  Test réel : POST /ai/extract-intent/audio (NestJS) ou --data-binary @f.wav :8014/transcribe"
	@echo ""

test-ai-image:
	@echo ""
	@code=$$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8015/health 2>/dev/null); \
	[ "$$code" = "200" ] \
	  && echo "  ✅ ai-vision:8015 opérationnel (SigLIP2 zero-shot)" \
	  || echo "  ❌ ai-vision:8015 non disponible → make logs-ai-vision"
	@echo "  Test réel : curl -X POST --data-binary @photo.jpg :8015/classify"
	@echo ""

test-api:
	@echo ""
	@echo "  Health :"; curl -s http://localhost:3000/health
	@echo ""
	@echo "  Swagger :"; curl -s -o /dev/null -w "  HTTP %{http_code}\n" http://localhost:3000/api/docs
	@echo ""

## ══════════════════════════════════════════════════════════════════════════════
## NGROK
## ══════════════════════════════════════════════════════════════════════════════

ngrok:
	@echo ""
	@echo "══════════════════════════════════════════════"
	@echo "  Tunnel ngrok — Domaine permanent"
	@echo "══════════════════════════════════════════════"
	@echo ""
	@if ! command -v ngrok &>/dev/null; then \
	  echo "  ngrok absent — installation..."; \
	  if [ -f /etc/debian_version ] || grep -qi ubuntu /etc/os-release 2>/dev/null; then \
	    curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
	      | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null 2>&1; \
	    echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
	      | sudo tee /etc/apt/sources.list.d/ngrok.list >/dev/null 2>&1; \
	    sudo apt-get update -qq 2>/dev/null && sudo apt-get install -y ngrok 2>/dev/null; \
	  elif command -v brew &>/dev/null; then \
	    brew install ngrok/ngrok/ngrok; \
	  else \
	    curl -sL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz \
	      | sudo tar xz -C /usr/local/bin; \
	  fi; \
	  echo "  ✅ ngrok installé !"; fi
	@NGROK_TOKEN=$$(grep '^NGROK_AUTH_TOKEN=' .env 2>/dev/null | cut -d= -f2- | tr -d ' "'); \
	if [ -z "$$NGROK_TOKEN" ]; then \
	  read -p "  Auth Token ngrok : " NGROK_TOKEN; \
	  grep -q '^NGROK_AUTH_TOKEN=' .env 2>/dev/null \
	    && $(SED_I) "s|^NGROK_AUTH_TOKEN=.*|NGROK_AUTH_TOKEN=$$NGROK_TOKEN|" .env \
	    || echo "NGROK_AUTH_TOKEN=$$NGROK_TOKEN" >> .env; fi; \
	ngrok config add-authtoken "$$(grep '^NGROK_AUTH_TOKEN=' .env | cut -d= -f2- | tr -d ' "')" >/dev/null 2>&1; \
	NGROK_DOMAIN=$$(grep '^NGROK_DOMAIN=' .env 2>/dev/null | cut -d= -f2- | tr -d ' "'); \
	if [ -z "$$NGROK_DOMAIN" ]; then \
	  read -p "  Domaine statique ngrok : " NGROK_DOMAIN; \
	  grep -q '^NGROK_DOMAIN=' .env 2>/dev/null \
	    && $(SED_I) "s|^NGROK_DOMAIN=.*|NGROK_DOMAIN=$$NGROK_DOMAIN|" .env \
	    || echo "NGROK_DOMAIN=$$NGROK_DOMAIN" >> .env; fi; \
	ND=$$(grep '^NGROK_DOMAIN=' .env | cut -d= -f2- | tr -d ' "'); \
	echo "  URL : https://$$ND"; \
	echo "  → Ctrl+C pour arrêter"; echo ""; \
	ngrok http --domain="$$ND" 80

ngrok-reset:
	@$(SED_I) '/^NGROK_AUTH_TOKEN=/d' .env 2>/dev/null
	@$(SED_I) '/^NGROK_DOMAIN=/d' .env 2>/dev/null
	@echo "✅ Config ngrok supprimée."

tunnel-quick:
	@cloudflared tunnel --url http://localhost:80 2>/dev/null \
	  || echo "cloudflared absent : sudo apt install cloudflared"

tunnel-stop:
	@pkill -f 'cloudflared tunnel' 2>/dev/null && echo "✅" || echo "Aucun tunnel."

tunnel-status:
	@ps aux | grep 'cloudflared tunnel' | grep -v grep || echo "Aucun tunnel."

flutter-run:
	@flutter run --dart-define=API_BASE_URL=http://$(LOCAL_IP):80

## ══════════════════════════════════════════════════════════════════════════════
## MINIO
## ══════════════════════════════════════════════════════════════════════════════

minio-buckets:
	@MK=$$(grep MINIO_ACCESS_KEY .env | cut -d= -f2 | tr -d '[:space:]'); \
	MS=$$(grep MINIO_SECRET_KEY .env | cut -d= -f2 | tr -d '[:space:]'); \
	docker run --rm --network khidmeti-network minio/mc:latest sh -c \
	  "mc alias set local http://minio:9001 $$MK $$MS && \
	   mc mb --ignore-existing local/profile-images && \
	   mc mb --ignore-existing local/service-media && \
	   mc mb --ignore-existing local/audio-recordings && \
	   mc anonymous set download local/profile-images && echo '✅ Buckets OK'"

minio-console:
	@$(OPEN_CMD) http://localhost:9002 2>/dev/null || echo "→ http://localhost:9002"

minio-list:
	@MK=$$(grep MINIO_ACCESS_KEY .env | cut -d= -f2 | tr -d '[:space:]'); \
	MS=$$(grep MINIO_SECRET_KEY .env | cut -d= -f2 | tr -d '[:space:]'); \
	docker run --rm --network khidmeti-network minio/mc:latest sh -c \
	  "mc alias set local http://minio:9001 $$MK $$MS && mc ls local/"

## ══════════════════════════════════════════════════════════════════════════════
## SCRIPTS
## ══════════════════════════════════════════════════════════════════════════════

scripts:
	@$(MAKE) scripts-migrations
	@$(MAKE) scripts-seeds

scripts-migrations:
	@echo ""
	@echo "  Migrations MongoDB"
	@MU=$$(grep '^MONGO_ROOT_USER' .env | cut -d= -f2 | tr -d '[:space:]'); \
	MP=$$(grep '^MONGO_ROOT_PASSWORD' .env | cut -d= -f2 | tr -d '[:space:]'); \
	C=0; F=0; shopt -s nullglob; FILES=(scripts/migrations/*.js); \
	if [ $${#FILES[@]} -eq 0 ]; then echo "  ⚪ Aucune migration."; \
	else \
	  for f in "$${FILES[@]}"; do \
	    echo "  → $$(basename $$f)"; \
	    docker exec -i khidmeti-mongo mongosh --quiet \
	      -u "$$MU" -p "$$MP" --authenticationDatabase admin khidmeti < "$$f" \
	    && { echo "  ✅"; C=$$((C+1)); } || { echo "  ❌"; F=$$((F+1)); }; \
	  done; echo "  ── $$C OK | $$F échec(s)"; [ $$F -gt 0 ] && exit 1 || true; fi
	@echo ""

scripts-seeds:
	@echo ""
	@echo "  Seeds TypeScript"
	@C=0; F=0; shopt -s nullglob; FILES=(apps/api/src/scripts/seeds/*.ts); \
	if [ $${#FILES[@]} -eq 0 ]; then echo "  ⚪ Aucun seed."; \
	else \
	  for f in "$${FILES[@]}"; do \
	    N=$$(basename "$$f"); echo "  → $$N $(ARGS)"; \
	    docker exec khidmeti-api npx ts-node --project tsconfig.json \
	      "src/scripts/seeds/$$N" $(ARGS) \
	    && { echo "  ✅"; C=$$((C+1)); } || { echo "  ❌"; F=$$((F+1)); }; \
	  done; echo "  ── $$C OK | $$F échec(s)"; [ $$F -gt 0 ] && exit 1 || true; fi
	@echo ""

scripts-%:
	@NAME=$*; \
	if [ -f "scripts/migrations/$$NAME.js" ]; then \
	  MU=$$(grep '^MONGO_ROOT_USER' .env | cut -d= -f2 | tr -d '[:space:]'); \
	  MP=$$(grep '^MONGO_ROOT_PASSWORD' .env | cut -d= -f2 | tr -d '[:space:]'); \
	  docker exec -i khidmeti-mongo mongosh --quiet \
	    -u "$$MU" -p "$$MP" --authenticationDatabase admin khidmeti \
	    < "scripts/migrations/$$NAME.js" \
	  && echo "✅ $$NAME OK" || exit 1; \
	elif [ -f "apps/api/src/scripts/seeds/$$NAME.ts" ]; then \
	  docker exec khidmeti-api npx ts-node --project tsconfig.json \
	    "src/scripts/seeds/$$NAME.ts" $(ARGS) \
	  && echo "✅ $$NAME OK" || exit 1; \
	elif [ -f "apps/api/src/scripts/$$NAME.ts" ]; then \
	  docker exec khidmeti-api npx ts-node --project tsconfig.json \
	    "src/scripts/$$NAME.ts" $(ARGS) \
	  && echo "✅ $$NAME OK" || exit 1; \
	else echo "❌ Script '$$NAME' introuvable."; exit 1; fi

## ══════════════════════════════════════════════════════════════════════════════
## SAUVEGARDE
## ══════════════════════════════════════════════════════════════════════════════

backup:
	@mkdir -p backups/mongodb/$(DATETIME)
	@MU=$$(grep MONGO_ROOT_USER .env | cut -d= -f2 | tr -d '[:space:]'); \
	MP=$$(grep MONGO_ROOT_PASSWORD .env | cut -d= -f2 | tr -d '[:space:]'); \
	docker exec khidmeti-mongo mongodump \
	  --username "$$MU" --password "$$MP" \
	  --authenticationDatabase admin --db khidmeti \
	  --out /tmp/bkp_$(DATETIME); \
	docker cp khidmeti-mongo:/tmp/bkp_$(DATETIME) backups/mongodb/$(DATETIME)
	@echo "✅ Sauvegarde → backups/mongodb/$(DATETIME)"

restore:
	@[ -n "$(BACKUP_DATE)" ] || { echo "Usage: make restore BACKUP_DATE=..."; exit 1; }
	@MU=$$(grep MONGO_ROOT_USER .env | cut -d= -f2 | tr -d '[:space:]'); \
	MP=$$(grep MONGO_ROOT_PASSWORD .env | cut -d= -f2 | tr -d '[:space:]'); \
	docker cp backups/mongodb/$(BACKUP_DATE) khidmeti-mongo:/tmp/rst; \
	docker exec khidmeti-mongo mongorestore \
	  --username "$$MU" --password "$$MP" \
	  --authenticationDatabase admin --db khidmeti --drop /tmp/rst/khidmeti
	@echo "✅ Restauration terminée."

## ══════════════════════════════════════════════════════════════════════════════
## SHELLS
## ══════════════════════════════════════════════════════════════════════════════

shell-api:
	@docker exec -it khidmeti-api /bin/sh

shell-mongo:
	@MU=$$(grep MONGO_ROOT_USER .env | cut -d= -f2 | tr -d '[:space:]'); \
	MP=$$(grep MONGO_ROOT_PASSWORD .env | cut -d= -f2 | tr -d '[:space:]'); \
	docker exec -it khidmeti-mongo mongosh -u "$$MU" -p "$$MP" \
	  --authenticationDatabase admin khidmeti

shell-redis:
	@RP=$$(grep REDIS_PASSWORD .env | cut -d= -f2 | tr -d '[:space:]'); \
	docker exec -it khidmeti-redis redis-cli -a "$$RP"

shell-minio:
	@MK=$$(grep MINIO_ACCESS_KEY .env | cut -d= -f2 | tr -d '[:space:]'); \
	MS=$$(grep MINIO_SECRET_KEY .env | cut -d= -f2 | tr -d '[:space:]'); \
	docker run -it --rm --network khidmeti-network minio/mc:latest \
	  sh -c "mc alias set local http://minio:9001 $$MK $$MS && sh"

shell-qdrant:
	@$(OPEN_CMD) http://localhost:6333/dashboard 2>/dev/null || echo "→ http://localhost:6333/dashboard"

mongo-stats:
	@MU=$$(grep MONGO_ROOT_USER .env | cut -d= -f2 | tr -d '[:space:]'); \
	MP=$$(grep MONGO_ROOT_PASSWORD .env | cut -d= -f2 | tr -d '[:space:]'); \
	docker exec khidmeti-mongo mongosh -u "$$MU" -p "$$MP" \
	  --authenticationDatabase admin khidmeti --quiet --eval "db.stats()"

redis-info:
	@RP=$$(grep REDIS_PASSWORD .env | cut -d= -f2 | tr -d '[:space:]'); \
	docker exec khidmeti-redis redis-cli -a "$$RP" INFO server

redis-flush:
	@RP=$$(grep REDIS_PASSWORD .env | cut -d= -f2 | tr -d '[:space:]'); \
	docker exec khidmeti-redis redis-cli -a "$$RP" FLUSHALL
	@echo "✅ Redis vidé."

## ══════════════════════════════════════════════════════════════════════════════
## NETTOYAGE
## ══════════════════════════════════════════════════════════════════════════════

clean-logs:
	@find logs/ -name "*.log" -delete 2>/dev/null || true
	@echo "✅ Logs nettoyés."

clean:
	@echo ""
	@echo "  ⚠️  Supprime volumes MongoDB, Redis, Qdrant, MinIO."
	@echo "  ✅ Les modèles dans docker/models/ sont CONSERVÉS."
	@echo ""
	@read -p "  Confirmer ? [y/N] " C; \
	[ "$$C" = "y" ] || [ "$$C" = "Y" ] \
	  && docker compose down -v --remove-orphans \
	  && docker system prune -f \
	  && echo "✅ Nettoyage terminé. Modèles intacts dans docker/models/" \
	  || echo "Annulé."

## ══════════════════════════════════════════════════════════════════════════════
## PRODUCTION  (docker-compose.prod.yml — image `production`, seul nginx:80 exposé)
## ══════════════════════════════════════════════════════════════════════════════

PROD_COMPOSE = docker compose -f docker-compose.prod.yml

## Vérification pré-prod : compile (tsc via nest build) + tests jest, dans le
## conteneur de dev (devDeps incluses). npm install s'exécute DANS Docker —
## rien à installer sur la machine hôte.
check:
	@echo "  🔍 Compilation + tests (dans Docker)..."
	@docker compose build api
	@docker compose run --rm --no-deps api sh -c "npm run build && npm run test"
	@echo "  ✅ check OK — build + tests passent."

## Promeut un compte au rôle admin (dashboard web). Le compte doit exister dans
## Firebase Auth ; le profil Mongo est créé automatiquement s'il n'existe pas.
##   make scripts-promote-admin ARGS="--email vous@example.com"
scripts-promote-admin:
	@docker compose run --rm --no-deps api \
	  npx ts-node --project tsconfig.json src/scripts/promote-admin.ts $(ARGS)

## Migration one-shot : chiffre les PII existantes (users + service_requests).
## Idempotente — relançable sans risque. À lancer UNE fois après prod-start.
encrypt-pii:
	@$(PROD_COMPOSE) run --rm --no-deps api node dist/scripts/encrypt-existing-pii.js

## Déploiement HuggingFace Spaces (hébergement MVP gratuit, remplace ngrok).
## Crée/met à jour les Spaces khidmeti-api + khidmeti-ai-embed, pousse le code,
## upsert les secrets depuis .env. Relançable = redéploiement.
hf-deploy:
	@python3 scripts/hf_deploy.py deploy

hf-status:
	@python3 scripts/hf_deploy.py status
	@echo "  ✅ PII chiffrées."

prod-start: _ensure-docker
	@$(PROD_COMPOSE) up -d --build --remove-orphans
	@echo "  ✅ Production démarrée."
	@if [ "$(NGROK)" != "0" ]; then \
	  echo ""; $(MAKE) --no-print-directory ngrok; \
	else \
	  echo "  (tunnel ngrok ignoré — NGROK=0)"; \
	fi

prod-stop:
	@$(PROD_COMPOSE) down --remove-orphans
	@echo "  ✅ Production arrêtée."

prod-update:
	@$(PROD_COMPOSE) build api
	@$(PROD_COMPOSE) up -d --no-deps api
	@echo "  ✅ API mise à jour."

prod-logs:
	@$(PROD_COMPOSE) logs --tail=100 -f

firewall:
	@echo "sudo ufw allow 80/tcp"
	@echo "# v15 : seul nginx:80 est exposé — 3000 et les ports IA sont internes au réseau Docker"
