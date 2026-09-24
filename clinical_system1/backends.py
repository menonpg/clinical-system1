"""Uniform `noul` interface over three local backends.

Each backend exposes  predict(state: str, questions: dict) -> dict[key, P(yes)]  and times itself.

- LayaBackend   : in-process laya.Router (ModernBERT / mmBERT encoders, prefill-only)
- KevBackend    : HTTP to a local `python -m kev.serve` (Qwen3.5 + LoRA + pointer head, TypeSafe System One API)
- OllamaBackend : a generative LLM asked one question at a time; P(yes) is read from the first-token
                  log-probabilities of "Yes" and "No" (renormalised), which is the strongest cheap baseline.
"""
from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request


class Backend:
    name = "base"

    def predict(self, state: str, questions: dict) -> dict[str, float]:
        raise NotImplementedError

    def timed(self, state: str, questions: dict) -> tuple[dict[str, float], float]:
        t = time.perf_counter()
        out = self.predict(state, questions)
        return out, (time.perf_counter() - t) * 1000


class LayaBackend(Backend):
    def __init__(self, model: str | None = None, device: str = "mps", max_len: int | None = None):
        from laya import Router
        self.router = Router(device=device)
        self.model, self.max_len = model, max_len
        self.name = f"laya-{model or 'router'}" + (f"-{max_len}" if max_len else "")

    def predict(self, state, questions):
        kw = {}
        if self.model:
            kw["model"] = self.model
        if self.max_len:
            kw["max_len"] = self.max_len
        r = self.router.predict(state, questions, **kw)
        return {k: float(v["noul"]) for k, v in r["answers"].items()}


class LayaFTBackend(Backend):
    """A locally fine-tuned Laya checkpoint directory (see experiments/finetune_laya.py)."""

    def __init__(self, path: str, device: str = "mps"):
        import laya
        self.agent = laya.load(path, device=device)
        self.name = "laya-ft"

    def predict(self, state, questions):
        r = self.agent.predict(state, questions)
        return {k: float(v["noul"]) for k, v in r["answers"].items()}


class KevBackend(Backend):
    def __init__(self, port: int, label: str):
        self.url = f"http://127.0.0.1:{port}/v1/systemone"
        self.name = label

    def predict(self, state, questions):
        body = json.dumps({"state": state, "model": "kev-latest", "questions": questions}).encode()
        for attempt in range(4):  # the Metal backend occasionally fails a command buffer under GPU contention
            try:
                req = urllib.request.Request(self.url, body, {"content-type": "application/json"})
                r = json.load(urllib.request.urlopen(req, timeout=600))
                break
            except urllib.error.HTTPError:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
        return {k: float(v["noul"]) for k, v in r["answers"].items()}


SYSTEM = ("You answer clinical screening questions about a patient record. "
          "Read the record carefully and reply with exactly one word: Yes or No.")


class OllamaBackend(Backend):
    def __init__(self, model: str = "gemma4:latest", host: str = "http://127.0.0.1:11434"):
        self.model, self.host = model, host
        self.name = f"ollama-{model.split(':')[0]}"

    def _one(self, state: str, question: str) -> float:
        body = {"model": self.model, "stream": False, "think": False, "logprobs": True, "top_logprobs": 10,
                "options": {"temperature": 0, "num_predict": 1, "num_ctx": 8192},
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": f"Record:\n{state}\n\nQuestion: {question}"}]}
        req = urllib.request.Request(f"{self.host}/api/chat", json.dumps(body).encode(), {"content-type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=600))
        yes = no = -math.inf
        for t in r["logprobs"][0]["top_logprobs"]:
            tok = t["token"].strip().lower()
            if tok == "yes":
                yes = max(yes, t["logprob"])
            elif tok == "no":
                no = max(no, t["logprob"])
        if yes == no == -math.inf:  # neither in top-k: fall back to the text
            return 1.0 if r["message"]["content"].strip().lower().startswith("y") else 0.0
        m = max(yes, no)
        return math.exp(yes - m) / (math.exp(yes - m) + math.exp(no - m))

    def predict(self, state, questions):
        return {k: self._one(state, q["instructions"]) for k, q in questions.items()}
