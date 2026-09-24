"""Derive an automation threshold per model from held-out results (used by the app).

For each model, rank lung-test notes by the confidence max(p, 1-p) of the checklist-AND score and find the
lowest confidence tau such that the auto-decided notes (confidence >= tau) have error <= budget.
Writes app/operating_points.json. The same recipe works on any small labelled set of your own.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clinical_system1.checklists import CHECKLISTS  # noqa: E402

RES = ROOT / "experiments" / "results"


def tau_for(y, p, budget):
    conf = np.maximum(p, 1 - p)
    o = np.argsort(-conf, kind="stable")
    err = ((p[o] >= .5).astype(int) != y[o]).astype(float)
    cum = np.cumsum(err) / np.arange(1, len(y) + 1)
    ok = np.where(cum <= budget)[0]
    if not len(ok):
        return 1.01, 0.0
    k = ok.max()
    return float(conf[o][k]), float((k + 1) / len(y))


def main():
    out = {}
    for split in ("lcs_test", "aaa_test"):
        ck = CHECKLISTS[split.split("_")[0]]
        for f in glob.glob(str(RES / f"e1_{split}_*.jsonl")):
            tag = Path(f).stem.split(f"{split}_", 1)[1]
            key = ("laya-ft" + ("+facts" if tag.endswith("+facts") else "")) if tag.startswith("laya-ft") else tag
            rows = [json.loads(l) for l in open(f)]
            if len(rows) < 50:
                continue
            y = np.array([r["labels"][f"{ck.key}_eligible"] for r in rows], int)
            p = np.array([ck.aggregate(r["p"]) for r in rows])
            ent = out.setdefault(key, {})
            for b in (0.05, 0.10):
                tau, cov = tau_for(y, p, b)
                ent[f"{ck.key}@{int(b * 100)}"] = {"tau": tau, "coverage": cov, "n": len(rows)}
    (ROOT / "app" / "operating_points.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
