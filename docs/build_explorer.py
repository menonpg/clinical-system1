"""Build docs/explorer.html (the Screening Explorer) with recorded predictions embedded as JSON."""
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from clinical_system1.checklists import CHECKLISTS  # noqa: E402
from clinical_system1.facts import augment  # noqa: E402
from run_checklists import dataset  # noqa: E402

RES = ROOT / "experiments" / "results"
NAMES = {"kev4b": "Kev-4B", "kev08b": "Kev-0.8B", "laya": "Laya EN (zero-shot)", "laya-ml": "Laya multilingual (zero-shot)",
         "laya-ft": "Laya EN fine-tuned (ours)", "ollama_gemma4_latest": "Gemma 4 (generative LLM)", "kev9b": "Kev-9B v2", "h2o4b": "H2O-Lightning-4B", "clefflash": "Cloudflare Clef-flash 9B", "kev27b": "Kev-27B v2 (Q4)"}
DS = {"hand": "40 hand-written notes", "test": "500 / 400 template notes (held-out templates)"}


def key(tag):
    base = tag.replace("+facts", "")
    base = "laya-ft" if base.startswith("laya-ft") else base
    return base + ("+facts" if tag.endswith("+facts") else "")


data = {"names": NAMES, "checklists": {}, "datasets": {}, "runs": {}}
for k, c in CHECKLISTS.items():
    data["checklists"][k] = {"key": k, "title": c.title, "criteria": [{"key": x.key, "question": x.question, "required": x.required} for x in c.criteria]}
data["dslabels"] = DS
for ds in DS:
    for ck in CHECKLISTS:
        rk = f"{ck}_{ds}"
        data["datasets"][rk] = {"notes": [{"id": r["id"], "text": r["text"], "labels": r["labels"],
                                           "facts": augment(r["text"])[len(r["text"]):].strip()} for r in dataset(rk)]}
        for f in glob.glob(str(RES / f"e1_{rk}_*.jsonl")):
            tag = key(Path(f).stem.split(f"{rk}_", 1)[1])
            rows = [json.loads(l) for l in open(f)]
            if len(rows) < 10:
                continue
            data["runs"].setdefault(rk, {})[tag] = {r["id"]: {k: round(v, 4) for k, v in r["p"].items()} for r in rows}
tpl = (ROOT / "docs" / "explorer_template.html").read_text()
out = ROOT / "docs" / "explorer.html"
out.write_text(tpl.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
print(out, f"{out.stat().st_size / 1e6:.2f} MB", {k: sorted(v) for k, v in data["runs"].items()})
