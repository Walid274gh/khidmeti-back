#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# KHIDMETI — seeds contre Atlas (ou toute MONGODB_URI cloud), depuis n'importe
# quelle machine avec node (Codespace, Termux…). Remplace le `docker exec
# khidmeti-api npx ts-node …` du Makefile qui suppose la stack locale.
#
# Usage :
#   export MONGODB_URI="mongodb+srv://user:pass@cluster.xxx.mongodb.net/khidmeti"
#   ./deploy/scripts/run-seeds.sh
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
: "${MONGODB_URI:?export MONGODB_URI=mongodb+srv://... d'abord}"

cd "$(dirname "$0")/../../apps/api"
[ -d node_modules ] || npm install --legacy-peer-deps

# Ordre voulu : professions d'abord (les workers y référencent).
for seed in src/scripts/seeds/seed-professions.ts src/scripts/seeds/seed-workers.ts; do
  echo "→ $seed"
  MONGODB_URI="$MONGODB_URI" npx ts-node --project tsconfig.json "$seed"
done
echo "✅ Seeds terminés."
