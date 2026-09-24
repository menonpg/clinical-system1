#!/bin/zsh
# E4: computed-facts hook on the lung checklist. Runs after run_all.sh finishes.
cd "$(dirname "$0")/.."
R=$PWD; E=$R/experiments; L=$R/upstream/laya/.venv/bin/python; LOG=$E/results
until grep -q "all done" $LOG/run_all.log; do sleep 30; done
kev_up() { (cd $R/upstream/kev && nohup uv run --extra serve python -m kev.serve --run $1 --port $2 > $LOG/kev_$2.log 2>&1 &); until curl -sf localhost:$2/v1/models >/dev/null; do sleep 5; done; }
kev_down() { pkill -f "kev.serve --run $1"; sleep 5; }
echo "== $(date +%T) facts start"
kev_up jaredpalmer/kev-4b 8009
python3 $E/run_checklists.py --backend kev4b --splits lcs_test,lcs_hand --facts
kev_down jaredpalmer/kev-4b
kev_up jaredpalmer/kev-0.8b 8019
python3 $E/run_checklists.py --backend kev08b --splits lcs_test,lcs_hand --facts
kev_down jaredpalmer/kev-0.8b
(cd /tmp && $L $E/run_checklists.py --backend laya-ft:$R/runs/laya-clinical --splits lcs_test,lcs_hand --facts)
(cd /tmp && $L $E/run_checklists.py --backend laya --splits lcs_test,lcs_hand --facts)
python3 $E/run_checklists.py --backend ollama:gemma4:latest --splits lcs_test --n 250 --facts
python3 $E/run_checklists.py --backend ollama:gemma4:latest --splits lcs_hand --facts
ollama stop gemma4:latest
(cd /tmp && $L $E/analyze.py > /dev/null && $L $E/figures.py)
echo "== $(date +%T) facts done"
