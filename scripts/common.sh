# Shared setup for local robot scripts. Source this: source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
source "$ROOT_DIR/../.venv/bin/activate"
source "$ROOT_DIR/env"

ROBOT_CAMERAS='{top: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: /dev/video2, width: 1280, height: 720, fps: 30, fourcc: MJPG}}'
