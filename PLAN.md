# Plan: local System-1 decision models for clinical checklists

Date: 2026-09-24

## Goal
Show that two open, local, non-generative decision models, **Laya** (NandhaKishorM/laya, ModernBERT/mmBERT,
322–421M) and **Kev** (jaredpalmer/kev, Qwen3.5 + LoRA + pointer head, 0.8B/4B), can do a useful job on
patient qualification. Compare them with a local generative LLM (Ollama) on accuracy, calibration,
latency and **safe automation** (the share of decisions that can be automated at a fixed error budget,
with the rest sent to a human).

Everything runs on one laptop (Apple M1 Max, 64 GB) and no patient data leaves the machine.

## Workspace layout
```
upstream/{laya,kev}      pristine clones (Apache-2.0)
data/                    public datasets (TREC CT 2021, generated guideline vignettes)
clinical_system1/        our package: checklist schemas, backends, aggregation, metrics
experiments/             runnable scripts + results JSON
docs/                    HTML documentation (upstream study + our project)
paper/                   arXiv LaTeX source + PDF
```

## Experiments
**E1: Standard-of-care checklist (USPSTF 2021 lung cancer screening, LDCT).**
The public criteria are age 50–80, ≥20 pack-years, and a current smoker or someone who quit within 15 years.
The public exclusions are health problems that substantially limit life expectancy, or unwillingness or inability to undergo curative surgery.
- Generate synthetic clinic notes from structured seeds. Ground truth comes exactly from the seeds. The notes vary how things are said: packs/day × years, cigarettes/day, quit year versus "quit N years ago", negation, distractors (a spouse who smokes, secondhand exposure), hospice or metastatic disease.
- Turn the checklist into typed questions: one `noul` per criterion plus a direct `noul` for "eligible".
- Build eligibility two ways: (a) end-to-end with the direct question, and (b) a checklist AND over the per-criterion probabilities (product rule), which gives an audit trail.
- Add a second checklist, USPSTF AAA screening (men 65–75 who ever smoked), to show the approach generalises.

**E2: Clinical-trial prescreening on a public benchmark (TREC Clinical Trials 2021).**
- Use the 75 public patient descriptions and ClinicalTrials.gov eligibility text. Take a balanced sample of judged pairs, eligible (2) versus excluded (1).
- Ask one direct `noul` question ("Does this patient meet the eligibility criteria?") with the criteria in the state. Also try a decomposed variant: an exclusion-criteria question plus an inclusion question.

**Baselines:** a local generative LLM through Ollama (gemma4, with a JSON yes/no answer and a verbalised confidence), plus majority/rule baselines.

**Metrics:** accuracy, balanced accuracy, AUROC, Brier, ECE, latency per decision, and coverage at a 5% and 10% error budget (selective prediction).

**Optional E3:** fine-tune Laya on training-template vignettes, then test on held-out templates to show that domain adaptation helps.

## Deliverables
1. HTML docs in `docs/`: the upstream study (Laya, Kev, comparison), the clinical project, and the results.
2. An open-source variant repo, `clinical-system1`, with the package, experiments and docs. **Published only after the user confirms.**
3. An arXiv-style paper (LaTeX to PDF) in `paper/`. **The user submits it to arXiv.**

## Caveats (stated in every deliverable)
Research prototype only. This is not a medical device and not clinical decision support for real patients.
The synthetic vignettes are not real EHR text, and TREC topics are also synthetic descriptions.
