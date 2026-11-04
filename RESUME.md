# Resume point (paused 2026-11-04 for travel)

Every runner skips items it has already finished, so restarting just continues.

## What is left
1. Clef-flash: TREC is 216 of 600 done; everything else is finished.
2. Kev-27B (Q4 GGUF, already downloaded): all splits.
3. Then: analysis, filling the red [TODO] markers in paper/main.tex (H2O, Clef-flash, Kev-27B), the abstract numbers,
   rewording the "Laya fine-tuned is the best" claims (H2O-4B + facts now matches it on hand-written AUROC),
   adding the note that the facts hook lowered H2O's life-limiting accuracy, the docs and explorer update, and the push.

## Restart (one command, about 2-3 h, detached)
```
cd ~/Documents/local-decision-models
nohup python3 -c "import os;os.setsid();os.execvp('zsh',['zsh','experiments/queue_new2.sh'])" > experiments/results/queue_new2.log 2>&1 &
```
Watch it with: `grep "^==" experiments/results/queue_new2.log`
