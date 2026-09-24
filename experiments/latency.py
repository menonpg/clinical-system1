"""Uncontended latency: the full LCS checklist (5 noul questions) per note, 50 notes, after 3 warm-up calls.

Run with nothing else on the GPU. Kev backends need their servers up; missing ones are skipped.
usage: upstream/laya/.venv/bin/python experiments/latency.py [--backends laya-mps,laya-cpu,laya-ft,kev08b,kev4b,gemma4]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from clinical_system1 import backends as B  # noqa: E402
from clinical_system1.checklists import LUNG  # noqa: E402
from run_checklists import dataset  # noqa: E402


def make(name):
    return {
        "laya-mps": lambda: B.LayaBackend(model="english", device="mps"),
        "laya-cpu": lambda: B.LayaBackend(model="english", device="cpu"),
        "laya-ft": lambda: B.LayaFTBackend(str(ROOT / "runs" / "laya-clinical"), device="mps"),
        "kev08b": lambda: B.KevBackend(8019, "kev-0.8b"),
        "kev4b": lambda: B.KevBackend(8009, "kev-4b"),
        "gemma4": lambda: B.OllamaBackend("gemma4:latest"),
    }[name]()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="laya-mps,laya-cpu,laya-ft,kev08b,kev4b,gemma4")
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    notes = [r["text"] for r in dataset("lcs_test")[: a.n]]
    q = LUNG.questions()
    out_f = ROOT / "experiments" / "results" / "latency.json"
    out = json.loads(out_f.read_text()) if out_f.exists() else {}
    for name in a.backends.split(","):
        try:
            be = make(name)
            for n in notes[:3]:
                be.predict(n, q)
        except Exception as e:  # server not running, checkpoint missing
            print("skip", name, e)
            continue
        ms = []
        for n in notes:
            t = time.perf_counter()
            be.predict(n, q)
            ms.append((time.perf_counter() - t) * 1000)
        out[name] = {"n_notes": len(ms), "questions_per_note": len(q), "p50_ms": float(np.median(ms)),
                     "p90_ms": float(np.percentile(ms, 90)), "mean_ms": float(np.mean(ms)),
                     "notes_per_s": 1000 / float(np.mean(ms)), "when": time.strftime("%Y-%m-%d %H:%M")}
        print(name, out[name], flush=True)
        out_f.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
