#!/bin/zsh
# Runs every experiment one model at a time (one model resident on the GPU), resuming finished items.
# Detach with:  python3 -c "import os;os.setsid();os.execvp('zsh',['zsh','experiments/run_all.sh'])" > experiments/results/run_all.log 2>&1 &
cd "$(dirname "$0")/.."
R=$PWD; E=$R/experiments; L=$R/upstream/laya/.venv/bin/python; LOG=$E/results
log() { echo "== $(date '+%H:%M:%S') $*"; }

kev_up() {  # $1 = hub id, $2 = port
  (cd $R/upstream/kev && nohup uv run --extra serve python -m kev.serve --run $1 --port $2 > $LOG/kev_$2.log 2>&1 &)
  until curl -sf localhost:$2/v1/models >/dev/null; do sleep 5; done
}
kev_down() { pkill -f "kev.serve --run $1"; sleep 5; }
laya() { (cd /tmp && $L "$@") }

log "kev-4b"
kev_up jaredpalmer/kev-4b 8009
python3 $E/run_checklists.py --backend kev4b --splits lcs_test,aaa_test,lcs_hand,aaa_hand
python3 $E/run_trec.py --backend kev4b
kev_down jaredpalmer/kev-4b

log "kev-0.8b"
kev_up jaredpalmer/kev-0.8b 8019
python3 $E/run_checklists.py --backend kev08b --splits lcs_test,aaa_test,lcs_hand,aaa_hand
python3 $E/run_trec.py --backend kev08b
kev_down jaredpalmer/kev-0.8b

log "laya zero-shot"
laya $E/run_checklists.py --backend laya --splits lcs_test,aaa_test,lcs_hand,aaa_hand
laya $E/run_checklists.py --backend laya-ml --splits lcs_test,aaa_test,lcs_hand,aaa_hand
laya $E/run_trec.py --backend laya-ml8k

log "laya fine-tune"
[ -f $R/runs/laya-clinical/model.safetensors ] || laya $E/finetune_laya.py --epochs 2 --out $R/runs/laya-clinical > $LOG/laya_ft.log 2>&1
laya $E/run_checklists.py --backend laya-ft:$R/runs/laya-clinical --splits lcs_test,aaa_test,lcs_hand,aaa_hand

log "gemma4"
python3 $E/run_checklists.py --backend ollama:gemma4:latest --splits lcs_test,aaa_test --n 250
python3 $E/run_checklists.py --backend ollama:gemma4:latest --splits lcs_hand,aaa_hand
python3 $E/run_trec.py --backend ollama:gemma4:latest --n 300
ollama stop gemma4:latest

log "latency (uncontended)"
laya $E/latency.py --backends laya-mps,laya-cpu,laya-ft
python3 -c "import os" && for m in "jaredpalmer/kev-0.8b 8019 kev08b" "jaredpalmer/kev-4b 8009 kev4b"; do
  set -- ${=m}; kev_up $1 $2; laya $E/latency.py --backends $3; kev_down $1
done
laya $E/latency.py --backends gemma4
ollama stop gemma4:latest

log "analysis"
laya $E/analyze.py > /dev/null && laya $E/figures.py
log "all done"
