#!/bin/bash
# Install (or remove) the Transcrb API as a macOS launchd agent.
#
#   ./install-service.sh              # install and start
#   ./install-service.sh --status
#   ./install-service.sh --uninstall
set -euo pipefail

LABEL="com.aliyoop.transcrb"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$HOME/.cache/transcrb"

while [ $# -gt 0 ]; do
  case "$1" in
    --uninstall)
      launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
      rm -f "$PLIST"
      echo "Removed $LABEL."
      exit 0 ;;
    --status)
      if launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; then
        echo "$LABEL is loaded."
        launchctl print "gui/$(id -u)/$LABEL" | grep -E '^\s+(state|pid|last exit code)' || true
        echo "--- health ---"
        curl -s -m 5 http://127.0.0.1:8756/api/health || echo "(not answering)"
        echo
        echo "--- resident memory ---"
        pid=$(launchctl print "gui/$(id -u)/$LABEL" | awk '/^\tpid = /{print $3}')
        [ -n "${pid:-}" ] && ps -o rss= -p "$pid" | awk '{printf "%.0f MB\n", $1/1024}'
        echo "--- recent log ---"
        tail -n 15 "$LOG_DIR/service.log" 2>/dev/null || echo "(no log yet)"
      else
        echo "$LABEL is NOT loaded."
      fi
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

# Preflight. Each of these has been an actual failure at least once.
[ -x "$REPO/.venv/bin/python" ] || { echo "Refusing: no .venv. Run the first-time setup." >&2; exit 1; }
grep -q '^inbox' "$REPO/config/config.toml" 2>/dev/null && {
  echo "Refusing: config/config.toml still has an inbox line. wrktbl-live owns filing now;" >&2
  echo "leaving it double-files every transcript into WorkBrain." >&2
  exit 1
}

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"
sed "s#/Users/aladdin/PROJECTS/Transcrb#$REPO#g" "$REPO/com.aliyoop.transcrb.plist" > "$PLIST"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
# bootout returns before the job is actually gone, and bootstrapping over a job
# that is still unloading fails with "Bootstrap failed: 5: Input/output error" —
# which leaves NO service running at all, because the old one did go away. Hit
# twice while fixing the capture throttle on 2026-08-05. Wait for it.
for _ in $(seq 50); do
  launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || break
  sleep 0.2
done
launchctl bootstrap "gui/$(id -u)" "$PLIST"

echo "Installed $LABEL on http://127.0.0.1:8756"
echo "  status:  ./install-service.sh --status"
echo "  log:     tail -f $LOG_DIR/service.log"
echo "  remove:  ./install-service.sh --uninstall"
