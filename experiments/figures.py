"""Figures: theme-aware SVG snippets for the docs (docs/figures/*.svg) and PDFs for the paper (paper/figures/*.pdf).

Reads experiments/results/summary.json and the per-item result files.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from clinical_system1.checklists import CHECKLISTS  # noqa: E402
from analyze import NAMES  # noqa: E402

RES = ROOT / "experiments" / "results"
DOCF = ROOT / "docs" / "figures"
PAPF = ROOT / "paper" / "figures"
S = json.loads((RES / "summary.json").read_text())
LAT = json.loads((RES / "latency.json").read_text()) if (RES / "latency.json").exists() else {}
ORDER = ["laya", "laya-ml", "laya-ft", "kev08b", "kev4b", "ollama_gemma4_latest"]
SHORT = {"laya": "Laya EN", "laya-ml": "Laya ML", "laya-ft": "Laya FT (ours)", "kev08b": "Kev-0.8B", "kev4b": "Kev-4B",
         "ollama_gemma4_latest": "Gemma 4 (gen.)", "laya-ml8k": "Laya ML 8k"}
# categorical slots (validated: light / dark), used by role
SER = [("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70"), ("#eda100", "#c98500")]


def ftag(t):
    return ("laya-ft" + ("+facts" if t.endswith("+facts") else "")) if t.startswith("laya-ft") else t


def e1(split):
    return {ftag(k): v for k, v in S["e1"].get(split, {}).items()}


# ------------------------------------------------------------------ SVG helpers (docs)
STYLE = """<style>
.viz{--s1:%s;--s2:%s;--s3:%s;--s4:%s}
@media (prefers-color-scheme: dark){:root:where(:not([data-theme="light"])) .viz{--s1:%s;--s2:%s;--s3:%s;--s4:%s}}
:root[data-theme="dark"] .viz{--s1:%s;--s2:%s;--s3:%s;--s4:%s}
.viz text{font-family:var(--sans);fill:var(--ink);font-size:12px}
.viz .ax{stroke:var(--rule);stroke-width:1}
.viz .grid{stroke:var(--rule);stroke-width:1;stroke-dasharray:2 3}
.viz .lab{fill:var(--ink-2);font-size:11px}
.viz rect.m:hover,.viz circle.m:hover{opacity:.8}
</style>""" % tuple([c[0] for c in SER] + [c[1] for c in SER] + [c[1] for c in SER])


def svg_wrap(w, h, body, label):
    return (f'<svg class="viz" viewBox="0 0 {w} {h}" role="img" aria-label="{label}">{STYLE}{body}</svg>')


def grouped_bars(groups, series, values, title, ymin=0.4, ymax=1.0, w=720, h=300, fmt="{:.2f}"):
    """groups: x labels; series: [(name, cssvar)]; values[g][s] (None = missing)."""
    L, R, T, B = 48, 16, 34, 46
    pw, ph = w - L - R, h - T - B
    y = lambda v: T + ph * (1 - (v - ymin) / (ymax - ymin))
    out = []
    for t in np.arange(ymin, ymax + 1e-9, 0.1):
        out.append(f'<line class="grid" x1="{L}" x2="{w - R}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>'
                   f'<text class="lab" x="{L - 6}" y="{y(t) + 4:.1f}" text-anchor="end">{t:.1f}</text>')
    gw = pw / len(groups)
    bw = min(26, (gw - 16) / len(series))
    for gi, g in enumerate(groups):
        x0 = L + gi * gw + (gw - bw * len(series) - 2 * (len(series) - 1)) / 2
        for si, (sn, var) in enumerate(series):
            v = values[gi][si]
            if v is None:
                continue
            x = x0 + si * (bw + 2)
            top = y(max(v, ymin))
            hgt = y(ymin) - top
            out.append(f'<path class="m" d="M{x:.1f},{y(ymin):.1f} V{top + 4:.1f} q0,-4 4,-4 h{bw - 8:.1f} q4,0 4,4 '
                       f'V{y(ymin):.1f} Z" fill="var({var})"><title>{g} · {sn}: {fmt.format(v)}</title></path>')
            out.append(f'<text x="{x + bw / 2:.1f}" y="{top - 5:.1f}" text-anchor="middle" font-size="10.5" '
                       f'class="lab">{fmt.format(v)}</text>')
        out.append(f'<text x="{L + gi * gw + gw / 2:.1f}" y="{h - B + 18}" text-anchor="middle">{g}</text>')
    out.append(f'<line class="ax" x1="{L}" x2="{w - R}" y1="{y(ymin)}" y2="{y(ymin)}"/>')
    lx = L
    for sn, var in series:
        out.append(f'<rect x="{lx}" y="10" width="11" height="11" rx="2" fill="var({var})"/>'
                   f'<text x="{lx + 16}" y="20">{sn}</text>')
        lx += 22 + 7.2 * len(sn)
    out.append(f'<text class="lab" x="{w - R}" y="20" text-anchor="end">{title}</text>')
    return svg_wrap(w, h, "".join(out), title)


def risk_coverage(curves, title, w=720, h=320, budget=(0.05, 0.10)):
    """curves: [(name, cssvar, coverage[], risk[])]"""
    L, R, T, B = 52, 120, 30, 44
    pw, ph = w - L - R, h - T - B
    ymax = 0.4
    x = lambda c: L + pw * c
    y = lambda r: T + ph * (1 - min(r, ymax) / ymax)
    out = []
    for t in np.arange(0, ymax + 1e-9, 0.1):
        out.append(f'<line class="grid" x1="{L}" x2="{L + pw}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>'
                   f'<text class="lab" x="{L - 6}" y="{y(t) + 4:.1f}" text-anchor="end">{t:.0%}</text>')
    for t in np.arange(0, 1.01, 0.25):
        out.append(f'<text class="lab" x="{x(t):.1f}" y="{T + ph + 16}" text-anchor="middle">{t:.0%}</text>')
    for b in budget:
        out.append(f'<line x1="{L}" x2="{L + pw}" y1="{y(b):.1f}" y2="{y(b):.1f}" stroke="var(--warn)" stroke-width="1" '
                   f'stroke-dasharray="5 4"/><text x="{L + 4}" y="{y(b) - 4:.1f}" class="lab">{b:.0%} error budget</text>')
    for name, var, cov, risk in curves:
        pts = " ".join(f"{x(c):.1f},{y(r):.1f}" for c, r in zip(cov, risk))
        out.append(f'<polyline points="{pts}" fill="none" stroke="var({var})" stroke-width="2"><title>{name}</title></polyline>')
        out.append(f'<text x="{x(cov[-1]) + 6:.1f}" y="{y(risk[-1]) + 4:.1f}">{name}</text>')
    out.append(f'<line class="ax" x1="{L}" x2="{L + pw}" y1="{T + ph}" y2="{T + ph}"/>')
    out.append(f'<text class="lab" x="{L + pw / 2}" y="{h - 6}" text-anchor="middle">coverage: share of notes decided automatically, '
               f'most confident first</text>')
    out.append(f'<text class="lab" x="12" y="{T - 12}">error rate among automated</text>')
    return svg_wrap(w, h, "".join(out), title)


def heat(rows, cols, vals, title, w=720):
    cw, rh, L, T = (w - 150) / len(cols), 30, 150, 44
    h = T + rh * len(rows) + 10
    out = []
    for ci, c in enumerate(cols):
        out.append(f'<text class="lab" x="{L + ci * cw + cw / 2:.1f}" y="{T - 10}" text-anchor="middle">{c}</text>')
    for ri, r in enumerate(rows):
        out.append(f'<text x="{L - 10}" y="{T + ri * rh + rh / 2 + 4}" text-anchor="end">{r}</text>')
        for ci, _ in enumerate(cols):
            v = vals[ri][ci]
            if v is None:
                continue
            a = max(0.0, (v - 0.5) / 0.5)  # 0.5 -> surface, 1.0 -> full accent
            out.append(f'<rect class="m" x="{L + ci * cw + 1:.1f}" y="{T + ri * rh + 1}" width="{cw - 2:.1f}" height="{rh - 2}" '
                       f'rx="3" fill="var(--s1)" fill-opacity="{0.08 + 0.85 * a:.2f}"><title>{r} · {cols[ci]}: {v:.3f}</title></rect>'
                       f'<text x="{L + ci * cw + cw / 2:.1f}" y="{T + ri * rh + rh / 2 + 4}" text-anchor="middle" '
                       f'style="fill:{"#fff" if a > .55 else "var(--ink)"}">{v:.2f}</text>')
    out.append(f'<text class="lab" x="12" y="16">{title}</text>')
    return svg_wrap(w, h, "".join(out), title)


def rc_curve(tag, split="lcs_test", mode="and"):
    f = [p for p in glob.glob(str(RES / f"e1_{split}_*.jsonl")) if ftag(Path(p).stem.split(f"{split}_", 1)[1]) == tag]
    if not f:
        return None
    ck = CHECKLISTS[split.split("_")[0]]
    rows = [json.loads(l) for l in open(f[0])]
    y = np.array([r["labels"][f"{ck.key}_eligible"] for r in rows], int)
    p = np.array([ck.aggregate(r["p"]) if mode == "and" else r["p"][f"{ck.key}_eligible"] for r in rows])
    conf = np.maximum(p, 1 - p)
    o = np.argsort(-conf, kind="stable")
    err = ((p[o] >= .5).astype(int) != y[o]).astype(float)
    n = np.arange(1, len(y) + 1)
    return list(n / len(y)), list(np.cumsum(err) / n)


def main():
    DOCF.mkdir(parents=True, exist_ok=True)
    PAPF.mkdir(parents=True, exist_ok=True)
    figs = {}
    for split, lab in (("lcs_test", "Lung cancer screening (USPSTF 2021)"), ("aaa_test", "AAA screening (USPSTF 2019)")):
        d = e1(split)
        tags = [t for t in ORDER if t in d]
        figs[f"auroc_{split}"] = grouped_bars(
            [SHORT[t] for t in tags], [("Direct question", "--s1"), ("Checklist-AND", "--s2")],
            [[d[t]["direct"]["auroc"], d[t]["checklist_and"]["auroc"]] for t in tags], f"{lab} · AUROC")
        figs[f"cov_{split}"] = grouped_bars(
            [SHORT[t] for t in tags], [("Direct question", "--s1"), ("Checklist-AND", "--s2")],
            [[d[t]["direct"]["cov@5%"], d[t]["checklist_and"]["cov@5%"]] for t in tags],
            f"{lab} · share automatable at ≤5% error", ymin=0.0, ymax=1.0)
        ck = CHECKLISTS[split.split("_")[0]]
        figs[f"crit_{split}"] = heat([SHORT[t] for t in tags], [c.key.split("_", 1)[1].replace("_", " ") for c in ck.criteria] + ["all correct"],
                                     [[d[t]["criteria"][c.key]["accuracy"] for c in ck.criteria] + [d[t]["all_criteria_correct"]] for t in tags],
                                     f"{lab} · per-criterion accuracy")
    curves = []
    for (tag, i) in (("laya", 0), ("laya-ft", 1), ("kev4b", 2), ("ollama_gemma4_latest", 3)):
        c = rc_curve(tag)
        if c:
            curves.append((SHORT[tag], f"--s{i + 1}", *c))
    figs["risk_lcs"] = risk_coverage(curves, "Risk–coverage, lung checklist-AND")
    t2 = [t for t in ["laya-ml8k", "kev08b", "kev4b", "ollama_gemma4_latest"] if t in S["e2"]]
    figs["trec"] = grouped_bars([SHORT[t] for t in t2], [("Direct question", "--s1"), ("Inclusion × (1 − exclusion)", "--s2")],
                                [[S["e2"][t]["direct"]["auroc"], S["e2"][t]["decomposed"]["auroc"]] for t in t2],
                                "TREC CT 2021 eligible vs excluded · AUROC", ymin=0.3, ymax=0.9)
    for k, v in figs.items():
        (DOCF / f"{k}.svg").write_text(v)
    paper(curves)
    print("figures:", ", ".join(figs))


def paper(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .3, "pdf.fonttype": 42})
    c1, c2 = SER[0][0], SER[1][0]
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.4), sharey=True)
    for ax, (split, lab) in zip(axs, (("lcs_test", "Lung (USPSTF 2021)"), ("aaa_test", "AAA (USPSTF 2019)"))):
        d = e1(split)
        tags = [t for t in ORDER if t in d]
        xs = np.arange(len(tags))
        a = [d[t]["direct"]["auroc"] for t in tags]
        b = [d[t]["checklist_and"]["auroc"] for t in tags]
        ax.bar(xs - .2, a, .38, color=c1, label="Direct question")
        ax.bar(xs + .2, b, .38, color=c2, label="Checklist-AND")
        for x_, v in zip(xs, b):
            ax.text(x_ + .2, v + .01, f"{v:.2f}", ha="center", fontsize=6)
        ax.set_xticks(xs, [SHORT[t] for t in tags], rotation=30, ha="right")
        ax.set_ylim(.4, 1.02)
        ax.set_title(lab)
    axs[0].set_ylabel("AUROC (eligible)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(PAPF / "auroc_e1.pdf")
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for (name, var, cov, risk), col in zip(curves, [c[0] for c in SER]):
        ax.plot(cov, risk, color=col, lw=1.5, label=name)
    for b in (.05, .10):
        ax.axhline(b, ls="--", lw=.8, color="#9a5a00")
    ax.set_ylim(0, .45)
    ax.set_xlabel("coverage (share auto-decided)")
    ax.set_ylabel("error rate among auto-decided")
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("Lung checklist-AND: risk–coverage")
    fig.tight_layout()
    fig.savefig(PAPF / "risk_coverage.pdf")
    if LAT:
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        d = e1("lcs_test")
        m = {"laya-mps": "laya", "laya-ft": "laya-ft", "kev08b": "kev08b", "kev4b": "kev4b", "gemma4": "ollama_gemma4_latest"}
        for lk, tag in m.items():
            if lk in LAT and tag in d:
                v = max(d[tag]["checklist_and"]["auroc"], d[tag]["direct"]["auroc"])
                ax.scatter(LAT[lk]["p50_ms"], v, s=22, color=c1, zorder=3)
                ax.annotate(SHORT[tag], (LAT[lk]["p50_ms"], v), textcoords="offset points", xytext=(4, 3), fontsize=6.5)
        ax.set_xscale("log")
        ax.set_xlabel("median ms per note, 5 questions (M1 Max)")
        ax.set_ylabel("best AUROC, lung")
        ax.set_title("Accuracy vs latency")
        fig.tight_layout()
        fig.savefig(PAPF / "latency.pdf")


if __name__ == "__main__":
    main()
