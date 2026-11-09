#!/bin/zsh
# Second stage: GGUF models served natively by llama.cpp, one at a time, after H2O finishes.
cd "$(dirname "$0")/.."
LOG=experiments/results
gguf() { ls ~/.cache/huggingface/hub/models--$1/snapshots/*/$2 2>/dev/null | head -1; }
wait_file() { until [ -n "$(gguf $1 $2)" ] && ! pgrep -f "hf_hub_download" >/dev/null; do sleep 30; done; }
LS="env GGML_METAL_BF16_DISABLE=1 llama-server -c 16384 -np 1 --no-kv-unified --no-mmproj --cache-ram 4096 -b 16384 -ub 16384"

wait_file ggml-org--Clef-Flash-GGUF Clef-Flash-Q8_0.gguf
experiments/bench_model.sh clefflash 8034 0 -- ${=LS} -m $(gguf ggml-org--Clef-Flash-GGUF Clef-Flash-Q8_0.gguf) --alias clef-flash --port 8034


wait_file kdercksen--Kev-27B-GGUF Kev-27B-Q4_K_M.gguf
experiments/bench_model.sh kev27b 8036 300 -- ${=LS} -m $(gguf kdercksen--Kev-27B-GGUF Kev-27B-Q4_K_M.gguf) --alias kev-27b --port 8036
echo "== queue2 all done $(date +%T)"
