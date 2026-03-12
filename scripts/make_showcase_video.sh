#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-debate}"
THEME="${2:-neon}"
QUESTION="${3:-Should an early startup pick monolith or microservices? Give a practical recommendation with tradeoffs.}"
OUT_PREFIX="${4:-showcase_debate}"

mkdir -p artifacts

CAST_PATH="artifacts/${OUT_PREFIX}.cast"
GIF_PATH="artifacts/${OUT_PREFIX}.gif"
MP4_PATH="artifacts/${OUT_PREFIX}.mp4"
POSTER_PATH="artifacts/${OUT_PREFIX}_poster.png"

CMD="python arena.py ${MODE} --theme ${THEME} --question \"${QUESTION}\""
if [[ "$MODE" != "showcase" ]]; then
  CMD="python arena.py ${MODE} --question \"${QUESTION}\""
fi

echo "Recording cast -> ${CAST_PATH}"
asciinema record \
  --headless \
  --overwrite \
  --window-size 170x48 \
  --idle-time-limit 1.2 \
  --command "zsh -lc 'cd ${ROOT_DIR} && ${CMD}'" \
  "${CAST_PATH}"

echo "Rendering GIF -> ${GIF_PATH}"
agg \
  --theme dracula \
  --font-size 20 \
  --idle-time-limit 1.0 \
  --speed 1.1 \
  "${CAST_PATH}" \
  "${GIF_PATH}"

echo "Converting MP4 -> ${MP4_PATH}"
ffmpeg -y -i "${GIF_PATH}" -movflags +faststart -pix_fmt yuv420p "${MP4_PATH}" >/dev/null 2>&1

echo "Extracting poster -> ${POSTER_PATH}"
ffmpeg -y -i "${MP4_PATH}" -ss 00:00:01 -vframes 1 "${POSTER_PATH}" >/dev/null 2>&1 || true

echo "Done."
ls -lh "${CAST_PATH}" "${GIF_PATH}" "${MP4_PATH}" "${POSTER_PATH}" 2>/dev/null || true
