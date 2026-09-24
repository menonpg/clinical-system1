# Clinical System One

**Apply a standard-of-care checklist to a clinical note on your own laptop.** Each criterion becomes a yes/no
question. A small open *decision model* ([Kev](https://github.com/jaredpalmer/kev) or
[Laya](https://github.com/NandhaKishorM/laya)) answers every question in one prefill pass and returns a calibrated
probability for each. A threshold measured on your own labelled notes then decides which cases are safe to
auto-screen and which go to a clinician.

> Research prototype. It is evaluated on synthetic and public benchmark text and is **not a medical device**.
> Do not use it for patient care without prospective validation.

## Try it in 3 minutes

```bash
git clone https://github.com/menonpg/clinical-system1 && cd clinical-system1
./setup.sh              # clones Laya + Kev into upstream/, creates both environments (needs uv)
app/run.sh              # starts Kev-4B on :8009 (first run downloads ~8 GB) and opens http://127.0.0.1:8765
```

In the app you can:

- **Screen a note.** Pick a checklist, or edit its criteria in the browser, then pick a model and an error budget.
  You get P(yes) per criterion, P(eligible), the criteria that failed, and one of three decisions:
  *auto-eligible*, *auto-ineligible* or *clinician review*.
- **Batch.** Paste many notes and get a decision table.
- **Use in your code.** Export the checklist as a System One HTTP request and as Python.

Nothing leaves the machine. With no GPU, `app/run.sh --no-kev` uses Laya on CPU or a local Ollama model instead.

**Screening Explorer** (`docs/explorer.html`, [hosted demo](https://claude.ai/artifact/9bpnn2XRzQ8XeoP9z3No5F)): recorded predictions from every model on the study notes. Drag
the automation threshold and watch coverage and error change, then open any note for its criterion-by-criterion audit.

## Do it for your own checklist (the recipe)

1. **One yes/no question per criterion.** State thresholds, inclusivity and units, and never combine two criteria
   in one question. Phrase exclusions positively and mark them `required=False`. See `clinical_system1/checklists.py`.
2. **Pick a model.** Kev-4B zero-shot if you have ≥16 GB unified memory or a GPU. Laya if you need CPU-only, but
   only after step 6.
3. **Give it the arithmetic.** `clinical_system1/facts.py` appends computed pack-years and years since quitting.
   Write the equivalent for your own units.
4. **Combine the answers.** P(eligible) = the product of the required answers. Every "no" names its criterion.
5. **Measure a threshold.** Label 50–200 of your own notes and run `experiments/operating_points.py`. It returns the
   confidence that keeps auto-decisions within your error budget and the share of notes that clears it.
6. **Fine-tune if coverage is too low.** Write a seed generator (`clinical_system1/vignettes.py`), train on one set
   of templates and test on another: `experiments/finetune_laya.py` takes about 90 minutes on an M1 Max.
7. **Deploy with review.** Add PHI redaction and audit hooks, have a clinician review a sample of automated
   negatives, and monitor coverage and error.

The paper (`paper/main.pdf`) walks through each step with the evidence behind it.

```python
from clinical_system1.checklists import LUNG
from clinical_system1.backends import KevBackend
from clinical_system1.facts import augment

kev = KevBackend(8009, "kev-4b")
note = "Visit 2026. 68 M, CAD s/p PCI 2019. Ex-smoker, 45 pk-yr hx, quit after his MI in 2019."
p = kev.predict(augment(note), LUNG.questions(include_direct=False))
print(LUNG.aggregate(p), LUNG.failed(p))      # P(eligible), criteria that block eligibility
```

## What's in the study

| | |
|---|---|
| **E1** | USPSTF 2021 lung-cancer and 2019 AAA screening. 900 notes from held-out templates with exact per-criterion labels, plus 40 hand-written notes |
| **E2** | TREC Clinical Trials 2021: 600 eligible-vs-excluded patient–trial pairs, with criteria from ClinicalTrials.gov |
| **E3** | Laya fine-tuned locally with its RLCD recipe, ported to a single Apple GPU |
| **E4** | The computed-facts hook, measured with and without |
| Models | Laya EN / multilingual, Kev-0.8B / 4B, and Gemma 4 through Ollama as a generative baseline, all on one M1 Max |

The numbers are in `experiments/results/summary.md` and on the docs site (`docs/index.html`).

## Layout

```
app/                 local screening app (FastAPI + one HTML page)
clinical_system1/    checklists, vignette generator, facts hook, backends, metrics
experiments/         data prep, runners, fine-tune, latency, analysis, figures, run_all.sh
data/vignettes/      generated notes + 40 hand-written notes (TREC data is fetched by experiments/prep_trec.py)
docs/                HTML docs: Laya, Kev, comparison, study design, results, explorer (python docs/build.py)
paper/               arXiv manuscript
```

## Credits and license

Apache-2.0. Built on [Laya](https://github.com/NandhaKishorM/laya) (Nandakishor M., Apache-2.0) and
[Kev](https://github.com/jaredpalmer/kev) (Jared Palmer, Apache-2.0), with Qwen3.5 base weights (Apache-2.0).
TREC Clinical Trials 2021 topics and judgements are from NIST, and trial text is from ClinicalTrials.gov.
USPSTF recommendations: Krist et al., JAMA 2021; Owens et al., JAMA 2019.
The Menon Lab, [themenonlab.blog](https://themenonlab.blog).
