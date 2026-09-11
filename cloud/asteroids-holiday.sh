#!/usr/bin/env bash
# Launch the holiday pipeline as a transient system unit so it outlives the SSH session, and
# arm the watchdog timer. Idempotent: refuses to start a second copy.
#
#   cloud/asteroids-holiday.sh              real run
#   cloud/asteroids-holiday.sh --smoke      minutes-long rehearsal of every step
#   cloud/asteroids-holiday.sh --fail-at r0 prove the fallback restarts asteroids.service
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
ROOT="$REPO/experiments/holiday-2026-09-11"
mkdir -p "$ROOT"

if systemctl is-active --quiet asteroids-holiday.service; then
  echo "asteroids-holiday.service is already running; stop it first" >&2
  exit 1
fi
sudo systemctl reset-failed asteroids-holiday.service 2>/dev/null || true

for unit in asteroids-holiday-watchdog.service asteroids-holiday-watchdog.timer; do
  sudo install -m 644 "cloud/$unit" "/etc/systemd/system/$unit"
done
sudo systemctl daemon-reload
sudo systemctl enable --now asteroids-holiday-watchdog.timer

sudo systemd-run --unit asteroids-holiday --uid ubuntu --gid ubuntu \
  -p WorkingDirectory="$REPO" \
  -p StandardOutput=append:"$ROOT/pipeline.log" \
  -p StandardError=append:"$ROOT/pipeline.log" \
  -E PYTHONPATH="$REPO/src" -E OMP_NUM_THREADS=1 -E MKL_NUM_THREADS=1 \
  "$REPO/.venv/bin/python" scripts/holiday_pipeline.py run "$@"
echo "started; follow with: tail -f $ROOT/pipeline.log"
