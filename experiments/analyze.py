"""Aggregate all result files into experiments/results/summary.json (plus a Markdown table)."""
from __future__ import annotations

import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from clinical_system1.checklists import CHECKLISTS  # noqa: E402
from clinical_system1.metrics import bootstrap_ci, summary, auroc  # noqa: E402
from run_checklists import dataset  # noqa: E402

RES = ROOT / "experiments" / "results"
NAMES = {"laya": "Laya (EN, zero-shot)", "laya-ml": "Laya-multilingual (zero-shot)", "laya-ml8k": "Laya-multilingual 8k",
         "kev4b": "Kev-4B", "kev08b": "Kev-0.8B", "ollama_gemma4_latest": "Gemma 4 (generative, logprob)",
         "laya-ft": "Laya fine-tuned (ours)", "kev08b-ft": "Kev-0.8B fine-tuned (ours)"}


def load(f):
    return [json.loads(l) for l in open(f)]


def lat(rows):
    ms = [r["ms"] for r in rows[1:]]  # first call includes model load / warm-up
    return {"p50_ms": float(np.median(ms)), "p90_ms": float(np.percentile(ms, 90))} if ms else {}


def e1():
    out = {}
    for f in sorted(glob.glob(str(RES / "e1_*_test_*.jsonl")) + glob.glob(str(RES / "e1_*_hand_*.jsonl"))):
        m = re.match(r"e1_(\w+?_(?:test|hand))_(.+)\.jsonl", Path(f).name)
        split, tag = m.group(1), m.group(2)
        if tag.startswith("laya-ft"):
            tag = "laya-ft" + ("+facts" if tag.endswith("+facts") else "")
        rows = load(f)
        if len(rows) < 2:
            continue
        ck = CHECKLISTS[split.split("_")[0]]
        full = len(dataset(split))
        meta = {r["id"]: r for r in dataset(split)}
        y = np.array([r["labels"][f"{ck.key}_eligible"] for r in rows], int)
        p_direct = np.array([r["p"][f"{ck.key}_eligible"] for r in rows])
        p_and = np.array([ck.aggregate(r["p"]) for r in rows])
        crit = {}
        for c in ck.criteria:
            yc = np.array([r["labels"][c.key] for r in rows], int)
            pc = np.array([r["p"][c.key] for r in rows])
            crit[c.key] = {"accuracy": float(((pc >= .5) == yc).mean()), "auroc": auroc(yc, pc), "prevalence": float(yc.mean())}
        # all criteria correct = audit trail fully right
        allc = np.mean([all((r["p"][c.key] >= .5) == r["labels"][c.key] for c in ck.criteria) for r in rows])
        # surface-form breakdown for the arithmetic criteria (LCS only)
        forms = {}
        if ck.key == "lcs":
            for key, attr in (("lcs_pack_years", "pack_form"), ("lcs_recency", "quit_form")):
                g = defaultdict(list)
                for r in rows:
                    if "seed" not in meta[r["id"]]:
                        continue
                    form = meta[r["id"]]["seed"][attr] or "n/a"
                    g[form].append((r["p"][key] >= .5) == r["labels"][key])
                forms[key] = {k: {"n": len(v), "accuracy": float(np.mean(v))} for k, v in g.items()}
        res = {
            "complete": len(rows) >= full, "n": len(rows),
            "direct": summary(y, p_direct), "checklist_and": summary(y, p_and),
            "direct_auroc_ci": bootstrap_ci(y, p_direct, auroc), "and_auroc_ci": bootstrap_ci(y, p_and, auroc),
            "criteria": crit, "all_criteria_correct": float(allc), "surface_forms": forms, "latency": lat(rows),
        }
        out.setdefault(split, {})[tag] = res
    return out


def e2():
    out = {}
    for f in sorted(glob.glob(str(RES / "e2_trec_*.jsonl"))):
        tag = Path(f).stem.replace("e2_trec_", "")
        rows = load(f)
        if len(rows) < 2:
            continue
        y = np.array([r["label"] for r in rows], int)
        pd = np.array([r["p"]["eligible"] for r in rows])
        pdec = np.array([r["p"]["meets_inclusion"] * (1 - r["p"]["meets_exclusion"]) for r in rows])
        out[tag] = {"n": len(rows), "complete": len(rows) >= 600, "direct": summary(y, pd), "decomposed": summary(y, pdec),
                    "direct_auroc_ci": bootstrap_ci(y, pd, auroc), "decomposed_auroc_ci": bootstrap_ci(y, pdec, auroc),
                    "latency": lat(rows)}
    return out


def md(s):
    L = []
    for split, d in s["e1"].items():
        L += [f"\n### E1 {split}\n", "| model | n | direct acc | direct AUROC | direct Brier | direct ECE | AND acc | AND AUROC | "
              "AND cov@5% | all-criteria-correct | p50 ms |", "|" + "---|" * 11]
        for tag, r in d.items():
            L.append(f"| {NAMES.get(tag, tag)} | {r['n']} | {r['direct']['accuracy']:.3f} | {r['direct']['auroc']:.3f} | "
                     f"{r['direct']['brier']:.3f} | {r['direct']['ece']:.3f} | {r['checklist_and']['accuracy']:.3f} | "
                     f"{r['checklist_and']['auroc']:.3f} | {r['checklist_and']['cov@5%']:.2f} | {r['all_criteria_correct']:.3f} | "
                     f"{r['latency'].get('p50_ms', float('nan')):.0f} |")
    L += ["\n### E2 TREC CT 2021 (eligible vs excluded)\n",
          "| model | n | direct acc | direct AUROC | direct Brier | decomposed acc | decomposed AUROC | p50 ms |", "|" + "---|" * 8]
    for tag, r in s["e2"].items():
        L.append(f"| {NAMES.get(tag, tag)} | {r['n']} | {r['direct']['accuracy']:.3f} | {r['direct']['auroc']:.3f} | "
                 f"{r['direct']['brier']:.3f} | {r['decomposed']['accuracy']:.3f} | {r['decomposed']['auroc']:.3f} | "
                 f"{r['latency'].get('p50_ms', float('nan')):.0f} |")
    return "\n".join(L)


if __name__ == "__main__":
    s = {"e1": e1(), "e2": e2()}
    (RES / "summary.json").write_text(json.dumps(s, indent=1))
    (RES / "summary.md").write_text(md(s))
    print(md(s))
