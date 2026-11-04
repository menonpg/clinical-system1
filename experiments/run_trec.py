"""E2: TREC Clinical Trials 2021 eligible-vs-excluded prescreening (resumable).

usage: python experiments/run_trec.py --backend laya-ml8k|kev4b|kev08b|ollama:gemma4:latest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_checklists import make_backend  # noqa: E402

MAX_CRIT = 6000  # chars per criteria block; keeps the longest records under ~4k tokens

QUESTIONS = {
    "eligible": {"type": "noul", "instructions":
                 "Based on the patient description, is this patient eligible for the clinical trial, "
                 "meaning they satisfy the inclusion criteria and meet none of the exclusion criteria?"},
    "meets_inclusion": {"type": "noul", "instructions":
                        "Does the patient satisfy the trial's inclusion criteria?"},
    "meets_exclusion": {"type": "noul", "instructions":
                        "Does the patient meet any of the trial's exclusion criteria?"},
}


def state(r: dict) -> str:
    age = f"{r['min_age'] or 'no minimum'} to {r['max_age'] or 'no maximum'}"
    return (f"PATIENT:\n{r['patient']}\n\nCLINICAL TRIAL: {r['title']}\nConditions: {', '.join(r['conditions'])}\n"
            f"Sex: {r['sex']}; Age: {age}\n\nINCLUSION CRITERIA:\n{r['inclusion'][:MAX_CRIT]}\n\n"
            f"EXCLUSION CRITERIA:\n{r['exclusion'][:MAX_CRIT] or 'None listed.'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    if a.backend == "laya-ml8k":
        from clinical_system1.backends import LayaBackend
        be = LayaBackend(model="multilingual", max_len=8192)
    else:
        be = make_backend(a.backend)
    rows = [json.loads(l) for l in open(ROOT / "data" / "trec_ct" / "sample.jsonl")]
    if a.n:  # the file is sorted by label, so take evenly spaced rows to keep the subset balanced
        step = len(rows) / a.n
        rows = [rows[int(i * step)] for i in range(a.n)]
    tag = a.backend.split(":", 2)[2] if a.backend.startswith("s1:") else a.backend.replace(":", "_")
    out = ROOT / "experiments" / "results" / f"e2_trec_{tag}.jsonl"
    done = {(json.loads(l)["topic"], json.loads(l)["nct"]) for l in open(out)} if out.exists() else set()
    with open(out, "a") as fh:
        for i, r in enumerate(rows):
            if (r["topic"], r["nct"]) in done:
                continue
            p, ms = be.timed(state(r), QUESTIONS)
            fh.write(json.dumps({"topic": r["topic"], "nct": r["nct"], "label": r["label"], "p": p, "ms": ms}) + "\n")
            fh.flush()
            if i % 50 == 0:
                print(f"[{be.name}] trec {i}/{len(rows)} {ms:.0f} ms", flush=True)


if __name__ == "__main__":
    main()
