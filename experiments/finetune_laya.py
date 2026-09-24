"""E3: fine-tune Laya on the checklist training split (template family A) on one Apple GPU.

This is a single-device port of Laya's published RLCD recipe
(upstream/laya/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb). The training loss is a
GRPO-style policy gradient on a strictly proper scoring-rule reward plus soft cross-entropy. A
temperature is then fitted per question type on a held-out calibration slice.
The test split (family B) uses different note templates and is never seen here.

usage: upstream/laya/.venv/bin/python experiments/finetune_laya.py --out runs/laya-clinical [--epochs 2]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, build_model, build_sequence, proper_reward

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
from clinical_system1.checklists import CHECKLISTS  # noqa: E402
from run_checklists import SPLITS, dataset  # noqa: E402


def collate(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax))
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, :k] = torch.tensor(it["target"])
    return ids, att, mpos, mmask, target, torch.tensor([it["qtype"] for it in items])


def fit_temp(pairs):
    Z = torch.tensor([z for z, _ in pairs])
    T = torch.tensor([t for _, t in pairs])
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.5, 5.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="convaiinnovations/laya")
    ap.add_argument("--out", default=str(ROOT / "runs" / "laya-clinical"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max_train", type=int, default=0, help="cap training questions (0 = all)")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.manual_seed(0)

    mdir = snapshot_download(a.base)
    _fix_tokenizer_config(mdir)
    tok = AutoTokenizer.from_pretrained(os.path.join(mdir, "tokenizer"))
    cfg = json.load(open(os.path.join(mdir, "rl_agent_config.json")))

    items = []
    for split in ("lcs_train", "aaa_train"):
        ck = CHECKLISTS[SPLITS[split][0]]
        for r in dataset(split):
            for key, q in ck.questions().items():
                y = float(r["labels"][key])
                seq, markers = build_sequence(tok, r["text"], {"t": "noul", "ins": q["instructions"], "crit": {}},
                                              cfg["max_len"], cfg["head_max_len"])
                items.append({"ids": seq, "markers": markers, "qtype": QTYPES["noul"], "target": [1 - y, y]})
    random.Random(20260924).shuffle(items)
    n_cal = min(1000, len(items) // 10)
    cal, train = items[:n_cal], items[n_cal:]
    if a.max_train:
        train = train[: a.max_train]
    print(f"{len(train)} train / {len(cal)} calibration questions", flush=True)

    model = build_model(cfg, encoder_dir=os.path.join(mdir, "encoder"))
    model.load_state_dict(load_file(os.path.join(mdir, "model.safetensors")), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(dev).train()

    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": 2.5e-5}, {"params": head, "lr": 1e-4}], weight_decay=0.01)
    steps = (len(train) // (a.micro * a.accum)) * a.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=1e-6)
    G, S0, S1 = 4, 0.4, 0.1
    t0 = time.time()
    for ep in range(a.epochs):
        random.Random(42 + ep).shuffle(train)
        sigma = S0 + (S1 - S0) * ep / max(1, a.epochs - 1)
        run, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for b in range(0, len(train), a.micro):
            ids, att, mpos, mmask, target, qt = [x.to(dev) for x in collate(train[b:b + a.micro], tok.pad_token_id)]
            logits, act = model(ids, att, mpos, mmask, qt)
            logits = logits.float()
            k = mmask.sum(-1, keepdim=True).float()
            eps = torch.randn((G,) + logits.shape, device=dev) * sigma * mmask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mmask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mmask, -1e4), -1)
            with torch.no_grad():
                rwd = proper_reward(q, target.unsqueeze(0), qt, mmask, w_sph=0.75, w_rps=1.0)
                adv = (rwd - rwd.mean(0, keepdim=True)) / ((rwd - rwd.mean(0, keepdim=True)).std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mmask).sum(-1) / (2 * sigma ** 2)
            loss = -(adv * logp).mean() - (target * torch.log_softmax(logits.masked_fill(~mmask, -1e4), -1)).sum(-1).mean()
            (loss / a.accum + 0.0 * act.sum()).backward()
            nb += 1
            run += loss.item()
            if nb % a.accum == 0 or b + a.micro >= len(train):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if nb % 100 == 0:
                print(f"ep {ep + 1} batch {nb}/{len(train) // a.micro} loss {run / nb:.4f} "
                      f"{time.time() - t0:.0f}s", flush=True)

    model.eval()
    pairs = []
    with torch.no_grad():
        for b in range(0, len(cal), 32):
            chunk = cal[b:b + 32]
            ids, att, mpos, mmask, target, qt = [x.to(dev) for x in collate(chunk, tok.pad_token_id)]
            lg, _ = model(ids, att, mpos, mmask, qt)
            for i, it in enumerate(chunk):
                pairs.append((lg[i, :2].float().cpu().tolist(), it["target"]))
    t_noul = fit_temp(pairs)
    acc = sum((z[1] > z[0]) == (t[1] > 0.5) for z, t in pairs) / len(pairs)
    print(f"calibration-slice accuracy {acc:.3f}; fitted noul temperature {t_noul:.3f}", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.float().contiguous().cpu() for k, v in model.state_dict().items()}, str(out / "model.safetensors"))
    for sub in ("encoder", "tokenizer"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
        shutil.copytree(os.path.join(mdir, sub), out / sub)
    temps = cfg.get("temperature", [1.0, 1.0, 1.0])
    temps = list(temps) if isinstance(temps, list) else [1.0, 1.0, 1.0]
    temps[QTYPES["noul"]] = t_noul
    cfg.update(fine_tuned=True, model_name="laya-clinical-checklists", temperature=temps)
    cfg.pop("temperature_by_options", None)
    json.dump(cfg, open(out / "rl_agent_config.json", "w"), indent=2)
    json.dump({"train_questions": len(train), "calibration_questions": len(cal), "epochs": a.epochs,
               "cal_accuracy": acc, "noul_temperature": t_noul, "seconds": time.time() - t0},
              open(out / "training_summary.json", "w"), indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
