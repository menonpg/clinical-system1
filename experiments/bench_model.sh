#!/bin/zsh
# Benchmark one local System One server on every split (E1 test + hand, both checklists; lung facts; TREC).
#   experiments/bench_model.sh <label> <port> <trec_n: 0=all 600> -- <serve command...>
# The serve command is started in the background, waited on until /v1/systemone answers, and stopped afterwards.
# Every runner resumes finished items, so re-running after an interruption is safe.
cd "$(dirname "$0")/.."
R=$PWD; E=$R/experiments; LOG=$E/results
LABEL=$1; PORT=$2; TRECN=$3; shift 4
echo "== $(date +%T) $LABEL: starting server on :$PORT"
(nohup "$@" > $LOG/serve_$LABEL.log 2>&1 &)
probe='{"state":"A 60-year-old man.","model":"m","questions":{"q":{"type":"noul","instructions":"Is the patient male?"}}}'
for i in {1..360}; do
  curl -sf -m 60 localhost:$PORT/v1/systemone -H 'content-type: application/json' -d $probe >/dev/null && break
  sleep 5
done
curl -sf -m 60 localhost:$PORT/v1/systemone -H 'content-type: application/json' -d $probe || { echo "== $LABEL: server never answered"; exit 1; }
echo
B="s1:$PORT:$LABEL"
python3 $E/run_checklists.py --backend $B --splits lcs_test,aaa_test,lcs_hand,aaa_hand
python3 $E/run_checklists.py --backend $B --splits lcs_test,lcs_hand --facts
python3 $E/run_trec.py --backend $B --n $TRECN
echo "== $(date +%T) $LABEL: done"
pkill -f -- "$*" ; lsof -ti tcp:$PORT | xargs kill 2>/dev/null; sleep 5
