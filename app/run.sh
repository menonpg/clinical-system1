#!/bin/zsh
# Start Kev-4B (optional) and the Clinical System One app, then open it.
#   app/run.sh            # app + Kev-4B
#   app/run.sh --no-kev   # app only (Laya / Ollama backends)
cd "$(dirname "$0")/.."
R=$PWD
NOKEV=0; [[ "$1" == "--no-kev" ]] && NOKEV=1 && shift
if [[ $NOKEV == 0 ]] && ! curl -sf localhost:8009/v1/models >/dev/null; then
  echo "starting Kev-4B on :8009 (first run downloads ~8 GB)"
  (cd upstream/kev && nohup uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009 > /tmp/kev4b.log 2>&1 &)
  until curl -sf localhost:8009/v1/models >/dev/null; do sleep 3; done
fi
[ -f app/operating_points.json ] || upstream/laya/.venv/bin/python experiments/operating_points.py >/dev/null
(sleep 3 && open http://127.0.0.1:8765) &
exec upstream/laya/.venv/bin/python app/server.py "$@"
