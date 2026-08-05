#!/usr/bin/env bash
# Build the native recorder.
#
# Recording falls back to ffmpeg without this, and ffmpeg's avfoundation input
# holds exactly one pending audio buffer and blocks the capture callback until it
# is read — about 10ms of tolerance for the whole capture. Measured on this
# machine under the macOS background throttle, ffmpeg lost 88.7% of a 30-second
# capture and the native recorder lost 0.1%. It is not an optimisation.
#
# Safe to run every time: it does nothing when the binary is already newer than
# the source, which is why run.sh can call it on every service start.
set -euo pipefail
cd "$(dirname "$0")/.."

SOURCE="native/capture.swift"
BINARY="native/transcrb-capture"

[ -f "$SOURCE" ] || { echo "missing $SOURCE" >&2; exit 1; }

if [ -x "$BINARY" ] && [ "$BINARY" -nt "$SOURCE" ]; then
  exit 0
fi

if ! command -v swiftc >/dev/null 2>&1; then
  cat >&2 <<'EOF'
! swiftc not found, so the native recorder cannot be built.
  Install the Xcode Command Line Tools and run this again:
      xcode-select --install
  Until then recording falls back to ffmpeg, which loses audio on a busy Mac.
EOF
  exit 1
fi

# Built to a temporary name and moved into place, so a failed compile never
# leaves a half-written binary that looks built and records nothing.
#
# Compiler output is held back unless the build fails. capture.swift calls
# AVCaptureDevice.devices(for:) on purpose — it is the list ffmpeg numbers its
# inputs from, and matching it is the whole point — and its deprecation warning
# printing during setup reads like something is wrong when nothing is.
if ! OUTPUT=$(swiftc -O -o "$BINARY.new" "$SOURCE" 2>&1); then
  echo "$OUTPUT" >&2
  rm -f "$BINARY.new"
  exit 1
fi
mv "$BINARY.new" "$BINARY"
echo "  ✓ native recorder built ($BINARY)"
