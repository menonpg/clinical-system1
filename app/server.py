"""Clinical System One: local screening app.

Paste a note, choose or build a checklist, pick a local model, and get:
  per-criterion P(yes) -> checklist-AND eligibility -> auto-decide or send to clinician review.

Run:  upstream/laya/.venv/bin/python app/server.py      (http://127.0.0.1:8765)
Kev backends need their servers running (see app/run.sh). Nothing leaves the machine.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from clinical_system1 import backends as B  # noqa: E402
from clinical_system1.checklists import CHECKLISTS  # noqa: E402
from clinical_system1.facts import augment  # noqa: E402

APP = Path(__file__).parent
OPS = json.loads((APP / "operating_points.json").read_text()) if (APP / "operating_points.json").exists() else {}
FT_DIR = ROOT / "runs" / "laya-clinical"
DEFAULT_TAU = 0.90  # used when no measured operating point exists (custom checklists, new models)

MODELS = {
    "kev4b": {"label": "Kev-4B", "kind": "kev", "port": 8009, "hub": "jaredpalmer/kev-4b"},
    "kev08b": {"label": "Kev-0.8B", "kind": "kev", "port": 8019, "hub": "jaredpalmer/kev-0.8b"},
    "laya-ft": {"label": "Laya fine-tuned (clinical)", "kind": "laya-ft"},
    "laya": {"label": "Laya EN (zero-shot)", "kind": "laya"},
    "gemma4": {"label": "Gemma 4 via Ollama (generative)", "kind": "ollama", "ops": "ollama_gemma4_latest"},
}
_loaded: dict[str, B.Backend] = {}
_lock = threading.Lock()


def available(key: str) -> tuple[bool, str]:
    m = MODELS[key]
    try:
        if m["kind"] == "kev":
            urllib.request.urlopen(f"http://127.0.0.1:{m['port']}/v1/models", timeout=1)
            return True, ""
        if m["kind"] == "laya-ft":
            return (FT_DIR / "model.safetensors").exists(), "fine-tuned checkpoint not found in runs/laya-clinical"
        if m["kind"] == "laya":
            return True, ""
        if m["kind"] == "ollama":
            tags = json.load(urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1))
            return any(t["name"].startswith("gemma4") for t in tags["models"]), "ollama pull gemma4"
    except Exception:
        pass
    if m["kind"] == "kev":
        return False, (f"start it: cd upstream/kev && uv run --extra serve python -m kev.serve "
                       f"--run {m['hub']} --port {m['port']}")
    return False, "Ollama is not running"


def backend(key: str) -> B.Backend:
    with _lock:
        if key not in _loaded:
            m = MODELS[key]
            _loaded[key] = {
                "kev": lambda: B.KevBackend(m.get("port", 0), m["label"]),
                "laya": lambda: B.LayaBackend(model="english", device=ARGS.device),
                "laya-ft": lambda: B.LayaFTBackend(str(FT_DIR), device=ARGS.device),
                "ollama": lambda: B.OllamaBackend("gemma4:latest"),
            }[m["kind"]]()
        return _loaded[key]


class Criterion(BaseModel):
    key: str
    question: str
    required: bool = True


class Checklist(BaseModel):
    key: str
    title: str
    direct: str | None = None
    criteria: list[Criterion]


class ScreenReq(BaseModel):
    note: str
    checklist: Checklist
    model: str
    facts: bool = False
    budget: int = 5  # error budget in percent for the operating point


class BatchReq(BaseModel):
    notes: list[str]
    checklist: Checklist
    model: str
    facts: bool = False
    budget: int = 5


def operating_point(model: str, ck_key: str, facts: bool, budget: int):
    ops_key = MODELS[model].get("ops", model) + ("+facts" if facts else "")
    op = OPS.get(ops_key, {}).get(f"{ck_key}@{budget}")
    if op:
        return op["tau"], f"measured on {op['n']} held-out notes: automates {op['coverage']:.0%} at ≤{budget}% error"
    return DEFAULT_TAU, f"no measured operating point for this model/checklist; default confidence ≥ {DEFAULT_TAU}"


def screen_one(note: str, ck: Checklist, model: str, facts: bool, budget: int) -> dict:
    text = augment(note) if facts else note
    qs = {c.key: {"type": "noul", "instructions": c.question} for c in ck.criteria}
    t = time.perf_counter()
    p = backend(model).predict(text, qs)
    ms = (time.perf_counter() - t) * 1000
    crit, p_and = [], 1.0
    for c in ck.criteria:
        pc = p[c.key]
        p_meet = pc if c.required else 1 - pc
        p_and *= p_meet
        crit.append({"key": c.key, "question": c.question, "required": c.required, "p_yes": pc, "p_meets": p_meet,
                     "meets": p_meet >= 0.5})
    tau, tau_note = operating_point(model, ck.key, facts, budget)
    conf = max(p_and, 1 - p_and)
    eligible = p_and >= 0.5
    if conf >= tau:
        decision = "auto_eligible" if eligible else "auto_ineligible"
    else:
        decision = "review"
    failed = [c["key"] for c in crit if not c["meets"]]
    weakest = min(crit, key=lambda c: max(c["p_meets"], 1 - c["p_meets"]))
    return {"criteria": crit, "p_eligible": p_and, "confidence": conf, "tau": tau, "tau_note": tau_note,
            "decision": decision, "failed": failed, "least_certain": weakest["key"], "latency_ms": ms,
            "augmented": text[len(note):].strip() if facts else ""}


app = FastAPI(title="Clinical System One")


@app.get("/")
def index():
    return FileResponse(APP / "static" / "index.html")


@app.get("/api/config")
def config():
    from run_checklists import dataset
    cks = [{"key": c.key, "title": c.title, "source": c.source, "direct": c.direct,
            "criteria": [{"key": x.key, "question": x.question, "required": x.required} for x in c.criteria]}
           for c in CHECKLISTS.values()]
    models = []
    for k, m in MODELS.items():
        ok, hint = available(k)
        models.append({"key": k, "label": m["label"], "available": ok, "hint": hint})
    examples = [{"id": r["id"], "text": r["text"],
                 "labels": {k: v for k, v in r["labels"].items()}} for r in dataset("lcs_hand")]
    return {"checklists": cks, "models": models, "examples": examples}


@app.post("/api/screen")
def screen(r: ScreenReq):
    ok, hint = available(r.model)
    if not ok:
        raise HTTPException(503, f"{MODELS[r.model]['label']} is not available. {hint}")
    if not r.note.strip():
        raise HTTPException(422, "Paste a note first.")
    return screen_one(r.note, r.checklist, r.model, r.facts, r.budget)


@app.post("/api/batch")
def batch(r: BatchReq):
    ok, hint = available(r.model)
    if not ok:
        raise HTTPException(503, f"{MODELS[r.model]['label']} is not available. {hint}")
    return [screen_one(n, r.checklist, r.model, r.facts, r.budget) | {"note": n} for n in r.notes if n.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="mps", help="mps | cuda | cpu (for Laya)")
    ARGS = ap.parse_args()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=ARGS.port)
