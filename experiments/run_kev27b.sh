#!/bin/zsh
# Kev-27B (Q4 GGUF) on a reduced set for time: 250 notes per template test set, all hand-written notes, 150 TREC pairs.
cd "$(dirname "$0")/.."
E=experiments; B="s1:8036:kev27b"
python3 $E/run_checklists.py --backend $B --splits lcs_test,aaa_test --n 250
python3 $E/run_checklists.py --backend $B --splits lcs_hand,aaa_hand
python3 $E/run_checklists.py --backend $B --splits lcs_test --n 250 --facts
python3 $E/run_checklists.py --backend $B --splits lcs_hand --facts
python3 $E/run_trec.py --backend $B --n 150
echo "== $(date +%T) kev27b: done"
pkill -f "llama-server.*8036"
