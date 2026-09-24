"""Write LaTeX tables for the paper from experiments/results/summary.json and latency.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
OUT = ROOT / "paper" / "tables"
OUT.mkdir(exist_ok=True)
S = json.loads((RES / "summary.json").read_text())
LAT = json.loads((RES / "latency.json").read_text()) if (RES / "latency.json").exists() else {}
ORDER = ["laya", "laya-ml", "laya-ft", "kev08b", "kev4b", "ollama_gemma4_latest"]
NAME = {"laya": "Laya EN (0-shot)", "laya-ml": "Laya ML (0-shot)", "laya-ft": "Laya EN + RLCD FT (ours)",
        "kev08b": "Kev-0.8B", "kev4b": "Kev-4B", "ollama_gemma4_latest": "Gemma~4 (generative)", "laya-ml8k": "Laya ML (8k ctx)"}
LATKEY = {"laya": "laya-mps", "laya-ft": "laya-ft", "kev08b": "kev08b", "kev4b": "kev4b", "ollama_gemma4_latest": "gemma4"}


def fit(tex):
    """Scale a table to the text width."""
    return tex.replace("\\begin{tabular}", "\\resizebox{\\linewidth}{!}{\\begin{tabular}", 1).replace("\\end{tabular}", "\\end{tabular}}", 1)


def norm(d):
    return {(("laya-ft" + ("+facts" if k.endswith("+facts") else "")) if k.startswith("laya-ft") else k): v for k, v in d.items()}


def best(vals, hi=True):
    v = [x for x in vals if x == x]
    return max(v) if hi else min(v)


def e1_table(split, caption, label):
    d = norm(S["e1"].get(split, {}))
    tags = [t for t in ORDER if t in d]
    cols = [("direct", "auroc", 1), ("checklist_and", "auroc", 1), ("checklist_and", "accuracy", 1),
            ("checklist_and", "brier", 0), ("checklist_and", "cov@5%", 1), ("checklist_and", "cov@10%", 1)]
    bests = [best([d[t][a][b] for t in tags], hi) for a, b, hi in cols] + [best([d[t]["all_criteria_correct"] for t in tags])]
    L = [r"\begin{table}[t]\centering\small", rf"\caption{{{caption}}}\label{{{label}}}",
         r"\begin{tabular}{lrrrrrrrr}\toprule",
         r" & Direct & \multicolumn{5}{c}{Checklist-AND} & All crit. & \\ \cmidrule(lr){2-2}\cmidrule(lr){3-7}",
         r"Model & AUROC & AUROC & Acc & Brier$\downarrow$ & Cov@5\% & Cov@10\% & correct & $n$ \\ \midrule"]
    for t in tags:
        r = d[t]
        cells = [r[a][b] for a, b, _ in cols] + [r["all_criteria_correct"]]
        fmt = [(rf"\textbf{{{v:.3f}}}" if abs(v - bv) < 1e-9 else f"{v:.3f}") for v, bv in zip(cells, bests)]
        L.append(f"{NAME[t]} & " + " & ".join(fmt) + f" & {r['n']} \\\\")
    L += [r"\bottomrule\end{tabular}\end{table}"]
    (OUT / f"{label.split(':')[1]}.tex").write_text(fit("\n".join(L)))


def crit_table():
    d = norm(S["e1"]["lcs_test"])
    tags = [t for t in ORDER if t in d]
    keys = ["lcs_age", "lcs_pack_years", "lcs_recency", "lcs_limiting"]
    L = [r"\begin{table}[t]\centering\small", r"\caption{Per-criterion accuracy on the lung-screening test set (500 notes, template family B).}\label{tab:crit}",
         r"\begin{tabular}{lrrrr}\toprule Model & Age 50--80 & $\geq$20 pack-yr & Recency & Limiting \\ \midrule"]
    for t in tags:
        L.append(NAME[t] + " & " + " & ".join(f"{d[t]['criteria'][k]['accuracy']:.3f}" for k in keys) + r" \\")
    L += [r"\bottomrule\end{tabular}\end{table}"]
    (OUT / "crit.tex").write_text("\n".join(L))
    # surface forms for pack-years
    L = [r"\begin{table}[t]\centering\small", r"\caption{Pack-year criterion accuracy by how the note states smoking history.}\label{tab:forms}",
         r"\begin{tabular}{lrrrr}\toprule Model & packs/day $\times$ yrs & cigarettes/day & stated pack-yrs & never smoker \\ \midrule"]
    for t in tags:
        f = d[t]["surface_forms"].get("lcs_pack_years", {})
        L.append(NAME[t] + " & " + " & ".join(f"{f[k]['accuracy']:.2f}" if k in f else "--" for k in ["ppd", "cigs", "packyears", "never"]) + r" \\")
    n = d[tags[0]]["surface_forms"]["lcs_pack_years"]
    L += [r"\midrule $n$ & " + " & ".join(str(n[k]["n"]) if k in n else "--" for k in ["ppd", "cigs", "packyears", "never"]) + r" \\",
          r"\bottomrule\end{tabular}\end{table}"]
    (OUT / "forms.tex").write_text("\n".join(L))


def trec_table():
    d = S["e2"]
    tags = [t for t in ["laya-ml8k", "kev08b", "kev4b", "ollama_gemma4_latest"] if t in d]
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{TREC Clinical Trials 2021, eligible vs.\ excluded (balanced; 95\% bootstrap CI for AUROC).}\label{tab:trec}",
         r"\begin{tabular}{lrrrrr}\toprule & \multicolumn{2}{c}{Direct} & \multicolumn{2}{c}{Inc.$\times$(1$-$Exc.)} & \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Model & AUROC & Bal.\ acc & AUROC & Bal.\ acc & $n$ \\ \midrule"]
    for t in tags:
        r = d[t]
        ci, ci2 = r["direct_auroc_ci"], r["decomposed_auroc_ci"]
        L.append(f"{NAME[t]} & {r['direct']['auroc']:.3f} [{ci[0]:.2f}, {ci[1]:.2f}] & {r['direct']['balanced_accuracy']:.3f} & "
                 f"{r['decomposed']['auroc']:.3f} [{ci2[0]:.2f}, {ci2[1]:.2f}] & {r['decomposed']['balanced_accuracy']:.3f} & {r['n']} \\\\")
    L += [r"\bottomrule\end{tabular}\end{table}"]
    (OUT / "trec.tex").write_text("\n".join(L))


def lat_table():
    rows = [("Laya EN, Apple GPU (MPS)", "laya-mps", "421M"), ("Laya EN, CPU only", "laya-cpu", "421M"),
            ("Laya FT (ours), MPS", "laya-ft", "421M"), ("Kev-0.8B, MLX bf16", "kev08b", "0.8B"),
            ("Kev-4B, MLX bf16", "kev4b", "4B"), ("Gemma~4, Ollama", "gemma4", "$\\sim$8B")]
    L = [r"\begin{table}[t]\centering\small", r"\caption{Latency for the full 5-question lung checklist per note on one Apple M1 Max, "
         r"uncontended (50 notes after 3 warm-up calls).}\label{tab:lat}",
         r"\begin{tabular}{llrrr}\toprule System & Params & p50 ms & p90 ms & notes/s \\ \midrule"]
    for name, k, p in rows:
        if k in LAT:
            r = LAT[k]
            L.append(f"{name} & {p} & {r['p50_ms']:.0f} & {r['p90_ms']:.0f} & {r['notes_per_s']:.2f} \\\\")
    L += [r"\bottomrule\end{tabular}\end{table}"]
    (OUT / "lat.tex").write_text("\n".join(L))


def facts_table():
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{Step 3 ablation: the computed-facts hook on the lung checklist (checklist-AND). $\Delta$ = with minus without.}\label{tab:facts}",
         r"\begin{tabular}{llrrrrr}\toprule Notes & Model & AUROC & Pack-yr acc & Recency acc & Cov@5\% & All crit. \\ \midrule"]
    for split, lab in (("lcs_test", "Template"), ("lcs_hand", "Hand-written")):
        d = norm(S["e1"].get(split, {}))
        for t in ["kev08b", "kev4b", "laya", "laya-ft", "ollama_gemma4_latest"]:
            if t in d and t + "+facts" in d:
                a, b = d[t], d[t + "+facts"]
                f = lambda r: [r["checklist_and"]["auroc"], r["criteria"]["lcs_pack_years"]["accuracy"],
                               r["criteria"]["lcs_recency"]["accuracy"], r["checklist_and"]["cov@5%"], r["all_criteria_correct"]]
                cells = [f"{y:.2f} ({y - x:+.2f})" for x, y in zip(f(a), f(b))]
                L.append(f"{lab} & {NAME[t]} & " + " & ".join(cells) + r" \\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule\end{tabular}\end{table}"
    (OUT / "facts.tex").write_text(fit("\n".join(L)))


if __name__ == "__main__":
    facts_table()
    e1_table("lcs_test", "USPSTF 2021 lung cancer screening eligibility on 500 held-out-template notes (40\\% eligible). "
             "Coverage = share of notes auto-decided at the given error budget. Bold = best.", "tab:lcs")
    e1_table("aaa_test", "USPSTF 2019 AAA screening eligibility on 400 held-out-template notes (40\\% eligible).", "tab:aaa")
    if "lcs_hand" in S["e1"]:
        e1_table("lcs_hand", "Lung screening on 40 hand-written free-text notes (13 eligible).", "tab:lcshand")
    if "aaa_hand" in S["e1"]:
        e1_table("aaa_hand", "AAA screening on the same 40 hand-written notes (10 eligible).", "tab:aaahand")
    crit_table()
    trec_table()
    lat_table()
    print("tables written:", sorted(p.name for p in OUT.iterdir()))


def questions_tex():
    import sys
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
    from clinical_system1.checklists import CHECKLISTS
    from run_trec import QUESTIONS
    esc = lambda s: s.replace("%", r"\%").replace("&", r"\&").replace("≥", r"$\geq$")
    L = [r"\begin{small}"]
    for ck in CHECKLISTS.values():
        L.append(rf"\paragraph{{{ck.title} ({esc(ck.source)}).}}\begin{{itemize}}")
        for c in ck.criteria:
            L.append(rf"\item \texttt{{{c.key.replace('_', chr(92) + '_')}}} (required: {'yes' if c.required else 'no'}): {esc(c.question)}")
        L.append(rf"\item \texttt{{{ck.key}\_eligible}} (direct): {esc(ck.direct)}\end{{itemize}}")
    L.append(r"\paragraph{TREC CT 2021.}\begin{itemize}")
    for k, q in QUESTIONS.items():
        L.append(rf"\item \texttt{{{k.replace('_', chr(92) + '_')}}}: {esc(q['instructions'])}")
    L += [r"\end{itemize}", r"\end{small}"]
    (ROOT / "paper" / "questions.tex").write_text("\n".join(L))


questions_tex()
