#!/bin/zsh
# Starts Enhance Speech. The first run sets up Python; the first file processed downloads the AI models (about 2 GB).
set -e
cd "$(dirname "$0")"
PORT="${PORT:-5840}"
if ! command -v ffmpeg >/dev/null; then
  echo "ffmpeg is required: brew install ffmpeg"; exit 1
fi
if [ ! -x .venv/bin/python ]; then
  command -v uv >/dev/null || { echo "uv is required: brew install uv"; exit 1; }
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -r requirements.txt
fi
if curl -fs "http://127.0.0.1:$PORT/api/tracks" >/dev/null 2>&1; then
  echo "Enhance Speech is already running: http://localhost:$PORT"
  [ -n "$NO_OPEN" ] || open "http://localhost:$PORT"
  exit 0
fi
echo "Enhance Speech: http://localhost:$PORT"
# Opens the page once the server answers. NO_OPEN=1 skips this.
if [ -z "$NO_OPEN" ]; then
  ( until curl -fs "http://127.0.0.1:$PORT/api/tracks" >/dev/null 2>&1; do sleep 1; done; open "http://localhost:$PORT" ) &
fi
exec .venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port "$PORT" --log-level warning
