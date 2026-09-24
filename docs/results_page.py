"""Generate docs/src/results.html from experiments/results/summary.json, latency.json and docs/figures/*.svg.

The narrative paragraphs live in docs/src/results_notes.html (hand-written after reading the numbers).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
FIG = ROOT / "docs" / "figures"
S = json.loads((RES / "summary.json").read_text())
LAT = json.loads((RES / "latency.json").read_text()) if (RES / "latency.json").exists() else {}
ORDER = ["laya", "laya-ml", "laya-ft", "kev08b", "kev4b", "ollama_gemma4_latest"]
NAME = {"laya": "Laya EN · zero-shot", "laya-ml": "Laya multilingual · zero-shot", "laya-ft": "Laya EN · fine-tuned (ours)",
        "kev08b": "Kev-0.8B", "kev4b": "Kev-4B", "ollama_gemma4_latest": "Gemma 4 · generative", "laya-ml8k": "Laya multilingual · 8k"}


def norm(d):
    return {(("laya-ft" + ("+facts" if k.endswith("+facts") else "")) if k.startswith("laya-ft") else k): v for k, v in d.items()}


def fig(name, cap):
    f = FIG / f"{name}.svg"
    return f"<figure>{f.read_text()}<figcaption>{cap}</figcaption></figure>" if f.exists() else ""


def e1_table(split):
    d = norm(S["e1"].get(split, {}))
    tags = [t for t in ORDER if t in d]
    cols = [("direct", "auroc", 1, "Direct AUROC"), ("checklist_and", "auroc", 1, "AND AUROC"),
            ("checklist_and", "accuracy", 1, "AND acc"), ("checklist_and", "brier", 0, "AND Brier ↓"),
            ("checklist_and", "cov@5%", 1, "Auto @5% err"), ("checklist_and", "cov@10%", 1, "Auto @10% err")]
    best = {c: (max if c[2] else min)(d[t][c[0]][c[1]] for t in tags) for c in cols}
    ac_best = max(d[t]["all_criteria_correct"] for t in tags)
    h = "<div class='tbl'><table><tr><th>Model</th>" + "".join(f"<th class='num'>{c[3]}</th>" for c in cols) + \
        "<th class='num'>All criteria right</th><th class='num'>n</th></tr>"
    for t in tags:
        r = d[t]
        cells = "".join(f"<td class='num{' best' if abs(r[c[0]][c[1]] - best[c]) < 1e-9 else ''}'>{r[c[0]][c[1]]:.3f}</td>" for c in cols)
        ac = r["all_criteria_correct"]
        h += f"<tr><td>{NAME[t]}</td>{cells}<td class='num{' best' if ac == ac_best else ''}'>{ac:.3f}</td><td class='num'>{r['n']}</td></tr>"
    return h + "</table></div>"


def trec_table():
    d = S["e2"]
    tags = [t for t in ["laya-ml8k", "kev08b", "kev4b", "ollama_gemma4_latest"] if t in d]
    h = ("<div class='tbl'><table><tr><th>Model</th><th class='num'>Direct AUROC [95% CI]</th><th class='num'>Direct bal. acc</th>"
         "<th class='num'>Inc×(1−Exc) AUROC [95% CI]</th><th class='num'>Inc×(1−Exc) bal. acc</th><th class='num'>n</th></tr>")
    for t in tags:
        r = d[t]
        a, b = r["direct_auroc_ci"], r["decomposed_auroc_ci"]
        h += (f"<tr><td>{NAME[t]}</td><td class='num'>{r['direct']['auroc']:.3f} [{a[0]:.2f}, {a[1]:.2f}]</td>"
              f"<td class='num'>{r['direct']['balanced_accuracy']:.3f}</td><td class='num'>{r['decomposed']['auroc']:.3f} "
              f"[{b[0]:.2f}, {b[1]:.2f}]</td><td class='num'>{r['decomposed']['balanced_accuracy']:.3f}</td><td class='num'>{r['n']}</td></tr>")
    return h + "</table></div>"


def facts_table():
    h = ("<div class='tbl'><table><tr><th>Notes</th><th>Model</th><th class='num'>AUROC without → with</th>"
         "<th class='num'>Pack-years acc</th><th class='num'>Recency acc</th><th class='num'>Auto @5% err</th></tr>")
    for split, lab in (("lcs_test", "Template"), ("lcs_hand", "Hand-written")):
        d = norm(S["e1"].get(split, {}))
        for t in ["kev08b", "kev4b", "laya", "laya-ft", "ollama_gemma4_latest"]:
            if t in d and t + "+facts" in d:
                a, b = d[t], d[t + "+facts"]
                g = lambda r, k: r["criteria"][k]["accuracy"]
                h += (f"<tr><td>{lab}</td><td>{NAME[t]}</td><td class='num'>{a['checklist_and']['auroc']:.3f} → <b>{b['checklist_and']['auroc']:.3f}</b></td>"
                      f"<td class='num'>{g(a,'lcs_pack_years'):.2f} → {g(b,'lcs_pack_years'):.2f}</td>"
                      f"<td class='num'>{g(a,'lcs_recency'):.2f} → {g(b,'lcs_recency'):.2f}</td>"
                      f"<td class='num'>{a['checklist_and']['cov@5%']:.2f} → {b['checklist_and']['cov@5%']:.2f}</td></tr>")
    return h + "</table></div>"


def lat_table():
    rows = [("Laya EN · Apple GPU", "laya-mps"), ("Laya EN · CPU only", "laya-cpu"), ("Laya fine-tuned · Apple GPU", "laya-ft"),
            ("Kev-0.8B · MLX", "kev08b"), ("Kev-4B · MLX", "kev4b"), ("Gemma 4 · Ollama", "gemma4")]
    h = "<div class='tbl'><table><tr><th>System</th><th class='num'>p50 ms / note</th><th class='num'>p90 ms</th><th class='num'>notes / s</th></tr>"
    for n, k in rows:
        if k in LAT:
            r = LAT[k]
            h += f"<tr><td>{n}</td><td class='num'>{r['p50_ms']:.0f}</td><td class='num'>{r['p90_ms']:.0f}</td><td class='num'>{r['notes_per_s']:.2f}</td></tr>"
    return h + "</table></div>"


notes = (ROOT / "docs" / "src" / "results_notes.html")
N = {}
if notes.exists():  # sections delimited by <!-- key --> markers
    cur = None
    for line in notes.read_text().splitlines():
        if line.startswith("<!-- ") and line.endswith(" -->"):
            cur = line[5:-4].strip()
            N[cur] = ""
        elif cur:
            N[cur] += line + "\n"

body = f"""<div class="eyebrow">Clinical study · results</div>
<h1>Checklist study results</h1>
{N.get('lede', '')}
<div class="note warn"><p><strong>Read with care.</strong> Synthetic and public benchmark notes only; nothing here is validated for patient care.</p></div>

<h2>E1 · Lung cancer screening (USPSTF 2021)</h2>
<p>500 held-out-template notes, 40% eligible. “AND” is the checklist-AND score. “Auto @5% err” is the share of notes that can be decided automatically, most confident first, with at most 5% errors among them. Bold marks the best value in each column.</p>
{e1_table('lcs_test')}
{fig('auroc_lcs_test', 'AUROC for eligibility: one direct question (blue) vs the product of per-criterion answers (orange). Hover a bar for its value.')}
{fig('risk_lcs', 'Risk–coverage for the lung checklist-AND score. A curve that stays under the dashed budget line longer automates more notes safely.')}
{N.get('lcs', '')}
<h3>Which criteria fail</h3>
{fig('crit_lcs_test', 'Per-criterion accuracy (0.5 threshold). The last column is the share of notes where every criterion is right, i.e. the audit trail is fully correct.')}
{N.get('crit', '')}

<h2>E1 · AAA screening (USPSTF 2019)</h2>
<p>400 held-out-template notes, 40% eligible.</p>
{e1_table('aaa_test')}
{fig('auroc_aaa_test', 'AAA eligibility AUROC, direct vs checklist-AND.')}
{N.get('aaa', '')}

<h2>E1b · 40 hand-written notes</h2>
<p>Free-text notes written by hand (abbreviations, indirect dates such as “quit after his MI in 2019”, boundary cases). 13 are lung-eligible and 10 AAA-eligible, so treat differences of a few points as noise.</p>
<h3>Lung</h3>
{e1_table('lcs_hand') if 'lcs_hand' in S['e1'] else '<p>Pending.</p>'}
<h3>AAA</h3>
{e1_table('aaa_hand') if 'aaa_hand' in S['e1'] else '<p>Pending.</p>'}
{N.get('hand', '')}

<h2>E4 · The computed-facts hook</h2>
<p>A deterministic pre-processor appends computed pack-years and years since quitting, and the model still makes every judgement. Values shown are checklist-AND on the lung checklist.</p>
{facts_table()}

<h2>E2 · Trial prescreening (TREC Clinical Trials 2021)</h2>
<p>Eligible vs excluded patient–trial pairs, balanced. Gemma 4 ran on a balanced 300-pair subset.</p>
{trec_table()}
{fig('trec', 'TREC CT 2021 AUROC: direct question vs inclusion × (1 − exclusion).')}
{N.get('trec', '')}

<h2>Latency on one laptop</h2>
<p>The whole 5-question lung checklist per note on an Apple M1 Max, with nothing else running (50 notes after warm-up).</p>
{lat_table()}
{N.get('latency', '')}

<h2>What this means</h2>
{N.get('takeaways', '')}
"""
(ROOT / "docs" / "src" / "results.html").write_text(body)
print("wrote docs/src/results.html")
