#!/bin/zsh
# One-time setup: clone the upstream models and create their environments. Needs git and uv (brew install uv).
set -e
cd "$(dirname "$0")"
mkdir -p upstream
LAYA_REV=970dc8c5f63d7b886a68409493f37d569424f933
KEV_REV=569ea449b1033c5aa8f149ec5e20d0c7a296d24a   # commits the study used
[ -d upstream/laya ] || { git clone https://github.com/NandhaKishorM/laya upstream/laya && git -C upstream/laya checkout -q $LAYA_REV; }
[ -d upstream/kev ]  || { git clone https://github.com/jaredpalmer/kev upstream/kev && git -C upstream/kev checkout -q $KEV_REV; }
uv venv --python 3.13 upstream/laya/.venv
uv pip install --python upstream/laya/.venv/bin/python -e "upstream/laya[serve]" numpy matplotlib
(cd upstream/kev && uv sync --extra serve)
echo "done. Start the app with: app/run.sh   (optional generative baseline: ollama pull gemma4)"
