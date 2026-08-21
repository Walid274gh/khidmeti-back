#!/data/data/com.termux/files/usr/bin/bash
# Poll les kernels Kaggle passés en argument, une ligne à CHAQUE changement d'état.
# Les flaps réseau (poll_error) ne sont signalés qu'après 3 tours tous en échec.
#   usage: poll_kernels.sh khidmeti-stt-ctc khidmeti-vision-mc2
cd /storage/emulated/0/opencode/khid-back/ml/kaggle || exit 1
[ $# -ge 1 ] || { echo "usage: poll_kernels.sh <slug> [slug...]"; exit 2; }

st() {
  python3 stt_push.py status "$1" 2>/dev/null | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
    s = d.get("status", "?")
    m = d.get("failureMessage") or ""
    print(s + ((" | " + m[:200]) if m else ""))
except Exception:
    print("poll_error")'
}

declare -A last
errs=0
while true; do
  bad=0; done_n=0
  for slug in "$@"; do
    s=$(st "$slug")
    if [ "$s" = "poll_error" ]; then
      bad=$((bad+1))
    else
      [ "$s" != "${last[$slug]}" ] && { echo "$slug: $s"; last[$slug]=$s; }
      case "$s" in complete*|error*|cancel*) done_n=$((done_n+1)) ;; esac
    fi
  done
  if [ "$bad" -eq "$#" ]; then
    errs=$((errs+1))
    [ "$errs" -eq 3 ] && echo "NETWORK: 3 tours de poll consécutifs en échec"
  else
    errs=0
  fi
  [ "$done_n" -eq "$#" ] && { echo "ALL_KERNELS_TERMINAL"; exit 0; }
  sleep 240
done
