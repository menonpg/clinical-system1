#!/bin/zsh
# Sequential queue for the new models (one GPU job at a time).
cd "$(dirname "$0")/.."
until grep -q "kev9b: done" experiments/results/bench_kev9b.log; do sleep 30; done
experiments/bench_model.sh h2o4b 8031 0 -- env GGML_METAL_BF16_DISABLE=1 llama-server -m /Users/prahladmenon/.cache/huggingface/hub/models--h2oai--h2o-lightning-4b-gguf/snapshots/61f8d8b7a8c2d84426d0d143ca187f65d235934b/h2o-lightning-4b-Q8_0.gguf -c 16384 -np 1 --no-kv-unified --no-mmproj --cache-ram 4096 --alias h2o-lightning-4b --port 8031
