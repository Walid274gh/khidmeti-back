# ══════════════════════════════════════════════════════════════════════════════
# KHIDMETI BACKEND — PowerShell Script v14.5 (parité Makefile)
# Usage:  .\khidmeti.ps1 [command] [args]
# Alias:  Set-Alias kh .\khidmeti.ps1
#
# WORKFLOW v15 — IA auto-hébergée (ai-nlu/ai-stt/ai-vision), état cloud :
#   .\khidmeti.ps1 start                  → TOUT : mode, modèles, services, ngrok
#   .\khidmeti.ps1 start cloud -NoNgrok   → bypass menu, sans tunnel
#   .\khidmeti.ps1 check                  → compile + tests (dans Docker)
#   .\khidmeti.ps1 prod-start             → compose production (nginx:80 seul)
#   .\khidmeti.ps1 encrypt-pii            → migration PII one-shot
# ══════════════════════════════════════════════════════════════════════════════
param(
  [Parameter(Position=0)]
  [string]$Command = "help",

  [Parameter(Position=1, ValueFromRemainingArguments)]
  [string[]]$ScriptArgs = @(),

  [switch]$NoNgrok
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PROD_COMPOSE = @("-f", "docker-compose.prod.yml")

# ── Couleurs ──────────────────────────────────────────────────────────────────
function Write-Header([string]$text) {
  Write-Host "`n══════════════════════════════════════════════" -ForegroundColor Cyan
  Write-Host "  $text" -ForegroundColor Cyan
  Write-Host "══════════════════════════════════════════════" -ForegroundColor Cyan
}
function Write-Ok([string]$msg)   { Write-Host "  ✅ $msg" -ForegroundColor Green  }
function Write-Warn([string]$msg) { Write-Host "  ⚠️  $msg" -ForegroundColor Yellow }
function Write-Err([string]$msg)  { Write-Host "  ❌ $msg" -ForegroundColor Red    }
function Write-Info([string]$msg) { Write-Host "  $msg"    -ForegroundColor Gray   }
function Write-Step([string]$msg) { Write-Host "  → $msg"  -ForegroundColor White  }

function Get-LocalIp {
  $ip = Get-NetIPAddress -AddressFamily IPv4 -PrefixOrigin Dhcp 2>$null |
    Where-Object { $_.IPAddress -match '^(192\.168|10\.|172\.(1[6-9]|2\d|3[01]))' } |
    Select-Object -First 1
  if ($ip) { return $ip.IPAddress }
  return "127.0.0.1"
}
$LOCAL_IP = Get-LocalIp

# ── Lire / écrire .env ────────────────────────────────────────────────────────
function Get-EnvValue([string]$key) {
  if (-not (Test-Path ".env")) { return "" }
  $line = Get-Content ".env" | Where-Object { $_ -match "^$key=" } | Select-Object -First 1
  if ($line) { return ($line -split "=", 2)[1].Trim().Trim('"') }
  return ""
}
function Set-EnvValue([string]$key, [string]$value) {
  if (-not (Test-Path ".env")) { New-Item -ItemType File -Path ".env" | Out-Null }
  $content = @(Get-Content ".env")
  if ($content | Where-Object { $_ -match "^$key=" }) {
    $content = $content -replace "^$key=.*", "$key=$value"
  } else {
    $content += "$key=$value"
  }
  $content | Set-Content ".env" -Encoding UTF8
}
function Remove-EnvValue([string]$key) {
  if (-not (Test-Path ".env")) { return }
  Get-Content ".env" | Where-Object { $_ -notmatch "^$key=" } |
    Set-Content ".env" -Encoding UTF8
}

# ── Prérequis ─────────────────────────────────────────────────────────────────
function Ensure-Docker {
  if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Err "Docker absent → https://docs.docker.com/desktop/install/windows-install/"
    exit 1
  }
  docker info 2>$null | Out-Null
  if ($LASTEXITCODE -ne 0) {
    Write-Err "Docker installé mais arrêté — lancez Docker Desktop puis réessayez."
    exit 1
  }
}

function Ensure-Dirs {
  @("logs", "backups\mongodb") |
    ForEach-Object {
      if (-not (Test-Path $_)) { New-Item -ItemType Directory -Path $_ -Force | Out-Null }
    }
}

# ── Mode cloud/local (parité _select-mode) ────────────────────────────────────
# .env est LE fichier unique (depuis v15) — éditez-le directement.
function Select-Mode([string]$requested) {
  $mode = $requested
  if ([string]::IsNullOrEmpty($mode)) { $mode = $env:MODE }
  if ([string]::IsNullOrEmpty($mode)) {
    Write-Host ""
    Write-Host "  ┌─────────────────────────────────────────────┐" -ForegroundColor White
    Write-Host "  │  Mode de démarrage Khidmeti                 │" -ForegroundColor White
    Write-Host "  │  1) cloud — Atlas / Upstash / Qdrant Cloud  │" -ForegroundColor White
    Write-Host "  │  2) local — mongo/redis/qdrant en Docker    │" -ForegroundColor White
    Write-Host "  └─────────────────────────────────────────────┘" -ForegroundColor White
    $c = Read-Host "  Choix [1-2] (défaut 1)"
    $mode = if ($c -eq "2" -or $c -eq "local") { "local" } else { "cloud" }
  }
  if ($mode -notin @("cloud", "local")) { Write-Err "Mode invalide : $mode (cloud|local)"; exit 1 }
  if (-not (Test-Path ".env.$mode"))    { Write-Err ".env.$mode introuvable"; exit 1 }
  Copy-Item ".env.$mode" ".env" -Force
  Write-Ok "Mode $mode → .env généré depuis .env.$mode"
}

function Test-Endpoint([string]$label, [string]$url) {
  try {
    $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5
    Write-Ok "$label — HTTP $($r.StatusCode)"
  } catch { Write-Err "$label — injoignable ($url)" }
}

# ── ngrok : installation + config + tunnel, tout automatique ──────────────────
function Ensure-Ngrok {
  if (Get-Command ngrok -ErrorAction SilentlyContinue) { return }
  Write-Step "ngrok absent — installation automatique…"
  $zipPath  = "$env:TEMP\ngrok.zip"
  $destPath = "C:\ngrok"
  Invoke-WebRequest -Uri "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip" `
    -OutFile $zipPath -UseBasicParsing
  if (-not (Test-Path $destPath)) { New-Item -ItemType Directory -Path $destPath -Force | Out-Null }
  Expand-Archive -Path $zipPath -DestinationPath $destPath -Force
  Remove-Item $zipPath -ErrorAction SilentlyContinue
  # PATH : session courante + persistance utilisateur (sans droits admin)
  $env:PATH += ";$destPath"
  $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
  if ($userPath -notlike "*$destPath*") {
    [Environment]::SetEnvironmentVariable("PATH", "$userPath;$destPath", "User")
  }
  Write-Ok "ngrok installé dans $destPath (ajouté au PATH)"
}

function Start-NgrokTunnel {
  Ensure-Ngrok
  $ngrokToken = Get-EnvValue "NGROK_AUTH_TOKEN"
  if ($ngrokToken -eq "") {
    Write-Info "https://dashboard.ngrok.com/get-started/your-authtoken"
    $ngrokToken = Read-Host "  Auth Token ngrok"
    Set-EnvValue "NGROK_AUTH_TOKEN" $ngrokToken
  }
  ngrok config add-authtoken $ngrokToken 2>$null | Out-Null
  $ngrokDomain = Get-EnvValue "NGROK_DOMAIN"
  if ($ngrokDomain -eq "") {
    Write-Info "https://dashboard.ngrok.com/domains"
    $ngrokDomain = Read-Host "  Domaine statique ngrok"
    Set-EnvValue "NGROK_DOMAIN" $ngrokDomain
  }
  Write-Host ""
  Write-Host "  URL : https://$ngrokDomain" -ForegroundColor Green
  Write-Host "  → Ctrl+C arrête le tunnel (les services restent up)" -ForegroundColor Gray
  Write-Host ""
  ngrok http "--domain=$ngrokDomain" 80
}

# ══════════════════════════════════════════════════════════════════════════════
# COMMANDES
# ══════════════════════════════════════════════════════════════════════════════
switch ($Command.ToLower()) {

  "help" {
    Write-Header "KHIDMETI v15 — IA auto-hébergée | IP : $LOCAL_IP"
    Write-Host ""
    Write-Host "  ⚠️  SETUP INITIAL :" -ForegroundColor Yellow
        Write-Info ".\khidmeti.ps1 start             TOUT : mode, modèles, services, ngrok"
    Write-Info "                                 bypass : start cloud|local ; sans tunnel : -NoNgrok"
    Write-Host ""
    Write-Host "  [Quotidien]" -ForegroundColor Cyan
    Write-Info "stop / restart / health / status / dns / logs / logs-api"
    Write-Info "build / rebuild                  Rebuild image api (dev)"
    Write-Host ""
    Write-Host "  [Production]" -ForegroundColor Cyan
    Write-Info "check                            Compile + tests jest (dans Docker)"
    Write-Info "prod-start                       Compose prod (image production, nginx:80 seul)"
    Write-Info "encrypt-pii                      Migration PII one-shot (après 1er prod-start)"
    Write-Info "prod-update / prod-stop / prod-logs"
    Write-Host ""
    Write-Host "  [Tests]" -ForegroundColor Cyan
    Write-Info "test-api / test-ai"
    Write-Host ""
    Write-Host "  [Tunnel]" -ForegroundColor Cyan
    Write-Info "ngrok / ngrok-reset              Tunnel permanent (installe ngrok si absent)"
    Write-Host ""
    Write-Host "  [Local mode]" -ForegroundColor Cyan
    Write-Info "logs-mongo / logs-redis / logs-qdrant / shell-mongo / shell-redis / backup"
    Write-Host ""
    Write-Info "clean-logs / clean / shell-api / flutter-run"
    Write-Host ""
  }

  # ── START : tout, de zéro jusqu'au tunnel ────────────────────────────────────
  "start" {
    Write-Header "Démarrage Khidmeti v15 — IA auto-hébergée"
    Ensure-Docker
    $requestedMode = if ($ScriptArgs.Count -gt 0) { $ScriptArgs[0] } else { "" }
    Select-Mode $requestedMode
    Write-Step "Démarrage des services…"
    docker compose up -d --remove-orphans
    Write-Host ""
    Write-Host ""
    & $PSCommandPath health
    & $PSCommandPath dns
    if (-not $NoNgrok) { Start-NgrokTunnel }
    else { Write-Info "(tunnel ngrok ignoré — -NoNgrok)" }
  }

  "stop" {
    docker compose down --remove-orphans
    Write-Ok "Services arrêtés. Modèles dans docker\models\ — intacts."
  }

  "restart" {
    docker compose down --remove-orphans
    Start-Sleep -Seconds 2
    & $PSCommandPath start @ScriptArgs
  }

  "build" {
    docker compose build --no-cache api
    Write-Ok "Build terminé."
  }

  "rebuild" {
    & $PSCommandPath build
    & $PSCommandPath start
  }

  # ── LOGS ─────────────────────────────────────────────────────────────────────
  "logs"          { docker compose logs --tail=100 -f }
  "logs-api"      { docker compose logs -f api }
  "logs-nginx"    { docker compose logs -f nginx }
  "logs-ai-embed" { docker compose logs -f ai-embed }
  "logs-mongo"    { docker compose logs -f mongo }
  "logs-redis"    { docker compose logs -f redis }
  "logs-qdrant"   { docker compose logs -f qdrant }

  # ── SANTÉ / ÉTAT ─────────────────────────────────────────────────────────────
  "health" {
    Write-Header "État des services"
    docker compose ps
    Write-Host ""
    Test-Endpoint "nginx  (:80)  " "http://localhost:80/health"
    Test-Endpoint "api    (:3000)" "http://localhost:3000/health"
    Write-Host ""
  }

  "status" { docker compose ps }

  "ai-status" {
    Write-Header "État IA"
    foreach ($d in @("nlu", "stt", "vision", "embed")) {
      $p = "docker\models\$d"
      if (Test-Path $p) { Write-Ok "$d — en cache" } else { Write-Warn "$d — absent (téléchargé au premier boot)" }
    }
    Write-Host ""
  }

  "dns" {
    Write-Header "URLs"
    Write-Info "API (nginx)  : http://${LOCAL_IP}:80"
    Write-Info "API (direct) : http://${LOCAL_IP}:3000"
    Write-Info "Swagger      : http://${LOCAL_IP}:3000/api/docs"
    $nd = Get-EnvValue "NGROK_DOMAIN"
    if ($nd -ne "") { Write-Info "ngrok        : https://$nd" }
    Write-Info "Flutter      : flutter run --dart-define=API_BASE_URL=http://${LOCAL_IP}:80"
    Write-Host ""
  }

  # ── MODÈLES ──────────────────────────────────────────────────────────────────

  # ── TESTS ────────────────────────────────────────────────────────────────────
  "test-api" {
    Test-Endpoint "api /health" "http://localhost:3000/health"
    Test-Endpoint "swagger    " "http://localhost:3000/api/docs"
  }

  "test-ai" {
    Write-Header "Test IA (ai-nlu) — Darija → JSON"
    $body = @{ text = "نحتاج بلومبي غدوا في وهران" } | ConvertTo-Json
    try {
      $r = Invoke-RestMethod -Uri "http://localhost:3000/ai/extract-intent" `
        -Method Post -ContentType "application/json" -Body $body -TimeoutSec 60
      $r | ConvertTo-Json -Depth 5
    } catch { Write-Err "Échec : $_" }
  }

  # ── PRODUCTION (parité Makefile) ─────────────────────────────────────────────
  "check" {
    Write-Header "Compilation + tests (dans Docker)"
    docker compose build api
    docker compose run --rm --no-deps api sh -c "npm run build && npm run test"
    if ($LASTEXITCODE -eq 0) { Write-Ok "check OK — build + tests passent." }
    else { Write-Err "check en échec."; exit 1 }
  }

  "scripts-promote-admin" {
    docker compose run --rm --no-deps api `
      npx ts-node --project tsconfig.json src/scripts/promote-admin.ts @ScriptArgs
  }

  "encrypt-pii" {
    docker compose @PROD_COMPOSE run --rm --no-deps api node dist/scripts/encrypt-existing-pii.js
    Write-Ok "PII chiffrées (users + service_requests). Relançable sans risque."
  }

  "prod-start" {
    Ensure-Docker
    docker compose @PROD_COMPOSE up -d --build --remove-orphans
    Write-Ok "Production démarrée."
    if (-not $NoNgrok) { Start-NgrokTunnel }
    else { Write-Info "(tunnel ngrok ignoré — -NoNgrok)" }
  }

  "prod-stop" {
    docker compose @PROD_COMPOSE down --remove-orphans
    Write-Ok "Production arrêtée."
  }

  "prod-update" {
    docker compose @PROD_COMPOSE build api
    docker compose @PROD_COMPOSE up -d --no-deps api
    Write-Ok "API mise à jour."
  }

  "prod-logs" { docker compose @PROD_COMPOSE logs --tail=100 -f }

  # ── NGROK ────────────────────────────────────────────────────────────────────
  "ngrok"         { Write-Header "Tunnel ngrok — Domaine permanent"; Start-NgrokTunnel }
  "ngrok-install" { Ensure-Ngrok; Write-Ok "ngrok prêt." }
  "ngrok-reset" {
    Remove-EnvValue "NGROK_AUTH_TOKEN"
    Remove-EnvValue "NGROK_DOMAIN"
    Write-Ok "Config ngrok supprimée."
  }

  # ── DIVERS ───────────────────────────────────────────────────────────────────
  "flutter-run" { flutter run --dart-define=API_BASE_URL="http://${LOCAL_IP}:80" }
  "shell-api"   { docker compose exec api sh }
  "shell-mongo" {
    $mu = Get-EnvValue "MONGO_ROOT_USER"; $mp = Get-EnvValue "MONGO_ROOT_PASSWORD"
    docker exec -it khidmeti-mongo mongosh -u $mu -p $mp --authenticationDatabase admin khidmeti
  }
  "shell-redis" { docker exec -it khidmeti-redis redis-cli }

  "backup" {
    Ensure-Dirs
    $mu = Get-EnvValue "MONGO_ROOT_USER"; $mp = Get-EnvValue "MONGO_ROOT_PASSWORD"
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    # Archive écrite DANS le conteneur puis docker cp — jamais de binaire via
    # un pipe PowerShell (corruption garantie en PS 5.1).
    docker exec khidmeti-mongo mongodump -u $mu -p $mp --authenticationDatabase admin `
      --db khidmeti --archive=/tmp/khidmeti.archive
    docker cp "khidmeti-mongo:/tmp/khidmeti.archive" "backups\mongodb\khidmeti-$stamp.archive"
    docker exec khidmeti-mongo rm /tmp/khidmeti.archive
    Write-Ok "Backup → backups\mongodb\khidmeti-$stamp.archive  (mode local uniquement)"
  }

  "clean-logs" {
    Get-ChildItem "logs" -File -ErrorAction SilentlyContinue | Remove-Item -Force
    Write-Ok "Logs nettoyés."
  }

  "clean" {
    docker compose down --remove-orphans -v
    Write-Ok "Volumes Docker supprimés (modèles dans docker\models\ — intacts)."
  }

  default {
    Write-Err "Commande inconnue : $Command"
    Write-Info ".\khidmeti.ps1 help"
    exit 1
  }
}
