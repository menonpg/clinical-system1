"""E1: USPSTF checklists over synthetic vignettes. Writes per-item predictions (resumable).

usage: python experiments/run_checklists.py --backend laya|kev4b|kev08b|gemma4 [--n 500]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clinical_system1.checklists import CHECKLISTS  # noqa: E402
from clinical_system1.vignettes import build  # noqa: E402

DATA = ROOT / "data" / "vignettes"
RES = ROOT / "experiments" / "results"
SPLITS = {  # (checklist, family, rng seed, n, positive fraction)
    "lcs_test": ("lcs", "B", 101, 500, 0.4),
    "aaa_test": ("aaa", "B", 202, 400, 0.4),
    "lcs_train": ("lcs", "A", 303, 3000, 0.4),
    "aaa_train": ("aaa", "A", 404, 2000, 0.4),
    "lcs_hand": ("lcs", None, None, 40, None),
    "aaa_hand": ("aaa", None, None, 40, None),
}


def dataset(split: str) -> list[dict]:
    DATA.mkdir(parents=True, exist_ok=True)
    if split.endswith("_hand"):  # 40 hand-written notes, labelled per criterion by the authors
        rows = [json.loads(l) for l in open(DATA / "handwritten.jsonl")]
        for r in rows:
            L = r["labels"]
            L["lcs_eligible"] = L["lcs_age"] and L["lcs_pack_years"] and L["lcs_recency"] and not L["lcs_limiting"]
            L["aaa_eligible"] = L["aaa_male"] and L["aaa_age"] and L["aaa_ever_smoked"]
        return rows
    f = DATA / f"{split}.jsonl"
    if not f.exists():
        ck, fam, seed, n, pos = SPLITS[split]
        rows = build(n, fam, seed, target=f"{ck}_eligible", pos_frac=pos)
        f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return [json.loads(l) for l in open(f)]


def make_backend(name: str):
    from clinical_system1 import backends as B
    if name == "laya":
        return B.LayaBackend(model="english")
    if name == "laya-ml":
        return B.LayaBackend(model="multilingual")
    if name.startswith("laya-ft:"):
        return B.LayaFTBackend(name.split(":", 1)[1])
    if name.startswith("s1:"):  # any local System One server: s1:<port>:<label>
        _, port, label = name.split(":", 2)
        return B.SystemOneBackend(f"http://127.0.0.1:{port}", label)
    if name == "kev4b":
        return B.KevBackend(8009, "kev-4b")
    if name == "kev08b":
        return B.KevBackend(8019, "kev-0.8b")
    if name.startswith("ollama:"):
        return B.OllamaBackend(name.split(":", 1)[1])
    raise SystemExit(f"unknown backend {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--splits", default="lcs_test,aaa_test")
    ap.add_argument("--n", type=int, default=0, help="limit items per split (0 = all)")
    ap.add_argument("--facts", action="store_true", help="append computed smoking arithmetic (clinical_system1.facts)")
    a = ap.parse_args()
    be = make_backend(a.backend)
    tag = (a.backend.split(":", 2)[2] if a.backend.startswith("s1:") else "laya-ft" if a.backend.startswith("laya-ft:") else a.backend.replace(":", "_").replace("/", "_")) + ("+facts" if a.facts else "")
    if a.facts:
        from clinical_system1.facts import augment
    for split in a.splits.split(","):
        ck = CHECKLISTS[SPLITS[split][0]]
        rows = dataset(split)[: a.n or None]
        out = RES / f"e1_{split}_{tag}.jsonl"
        done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
        with open(out, "a") as fh:
            for i, r in enumerate(rows):
                if r["id"] in done:
                    continue
                p, ms = be.timed(augment(r["text"]) if a.facts else r["text"], ck.questions())
                fh.write(json.dumps({"id": r["id"], "p": p, "ms": ms, "labels": r["labels"]}) + "\n")
                fh.flush()
                if i % 50 == 0:
                    print(f"[{be.name}] {split} {i}/{len(rows)} {ms:.0f} ms", flush=True)


if __name__ == "__main__":
    main()
