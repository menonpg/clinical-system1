# arXiv submission metadata

Paste-ready fields for the arXiv metadata form. The abstract is plain text with no LaTeX, 1,889 characters (limit 1,920).

## Title

```
Checklists at the Speed of Prefill: Local, Calibrated Clinical Eligibility Screening with Open Decision Models
```

## Author(s)

```
Prahlad G. Menon
```

## Abstract

```
Much of standard-of-care qualification means applying a published checklist to a free-text chart. We ask whether open decision models, which answer typed yes/no questions with calibrated probabilities from a single prefill pass, can do this on one laptop, and which practical steps make them safe to use. We encoded the USPSTF 2021 lung-cancer and 2019 abdominal aortic aneurysm screening recommendations as one question per criterion, combined into an auditable conjunctive eligibility score. We evaluated ten local systems (nine decision models from four families released September-November 2026, and one generative LLM) on 900 synthetic notes from held-out templates with exact per-criterion labels, 40 hand-written notes, and 600 eligible-versus-excluded patient-trial pairs from TREC Clinical Trials 2021. Newer open decision models clearly outperformed earlier ones: lung AUROC on template notes was 0.977 for Kev-27B and 0.976 for Clef-flash 9B, against 0.877 for Kev-4B; Laya was near chance. Criteria requiring arithmetic were the main source of error. A deterministic hook stating pack-years and years since quitting raised hand-written-note AUROC to 0.997-1.000 for the strongest models, and Kev-27B could then auto-decide 95% of hand-written notes at <=5% error. An 85-minute on-laptop fine-tune lifted Laya from 0.55 to 0.96 with the hook, at 124 ms per note. On trial prescreening, 4B-27B decision models reached AUROC 0.82-0.90, above an 8B generative baseline (0.80). Automation thresholds measured on synthetic notes overstated safe coverage on hand-written notes by up to 2.6-fold. Open decision models can screen checklists locally, with no note leaving the machine, if the arithmetic is made explicit and the automation threshold is measured on the practitioner's own notes. We release a step-by-step guide, a local screening app, data generators and all predictions.
```

## Comments

```
19 pages, 3 figures, 9 tables. Code, screening app, data generators and per-item predictions: https://github.com/menonpg/clinical-system1
```

## ACM class

```
I.2.7; J.3
```

I.2.7 is Natural Language Processing; J.3 is Life and Medical Sciences.

## Leave blank

- Report number
- Journal reference
- External DOI
- MSC class

## Categories

- **Primary (for now):** cs.ET. Change it to **cs.AI** once you are endorsed (code SRL6WD).
- **Cross-lists:** cs.CL, cs.LG, cs.AI
