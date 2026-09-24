"""Public screening checklists written as typed System-One questions.

Each criterion is a `noul` (yes/no) question with a required polarity. Eligibility can be read
two ways: from the direct question, or as an audited AND over the criteria:

    P(eligible) = prod_i  p_i      if criterion i must be "yes"
                          (1-p_i)  if criterion i must be "no"

This assumes the criteria are conditionally independent given the note, so it is a heuristic.
In return, every "no" can be traced back to the criterion that failed.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Criterion:
    key: str
    question: str
    required: bool = True  # the answer that eligibility requires


@dataclass(frozen=True)
class Checklist:
    key: str
    title: str
    source: str
    direct: str
    criteria: tuple[Criterion, ...] = field(default_factory=tuple)

    def questions(self, include_direct: bool = True) -> dict[str, dict]:
        q = {c.key: {"type": "noul", "instructions": c.question} for c in self.criteria}
        if include_direct:
            q[f"{self.key}_eligible"] = {"type": "noul", "instructions": self.direct}
        return q

    def aggregate(self, p: dict[str, float]) -> float:
        out = 1.0
        for c in self.criteria:
            out *= p[c.key] if c.required else 1.0 - p[c.key]
        return out

    def failed(self, p: dict[str, float], thr: float = 0.5) -> list[str]:
        """Criteria that block eligibility at threshold thr (the audit trail)."""
        return [c.key for c in self.criteria if (p[c.key] >= thr) != c.required]


LUNG = Checklist(
    key="lcs",
    title="Lung cancer screening with low-dose CT",
    source="USPSTF 2021 (Grade B), JAMA 2021;325(10):962-970",
    direct=("Under the USPSTF 2021 lung cancer screening recommendation (adults aged 50 to 80 years who have a "
            "20 pack-year smoking history and currently smoke or have quit within the past 15 years, and who do not "
            "have a health problem that substantially limits life expectancy or the ability or willingness to have "
            "curative lung surgery), is this patient eligible for annual low-dose CT screening?"),
    criteria=(
        Criterion("lcs_age", "Is the patient between 50 and 80 years old, inclusive?"),
        Criterion("lcs_pack_years", "Does the patient have a smoking history of at least 20 pack-years "
                                    "(packs per day multiplied by years smoked; 20 cigarettes = 1 pack)?"),
        Criterion("lcs_recency", "Does the patient currently smoke, or did they quit smoking 15 or fewer years ago?"),
        Criterion("lcs_limiting", "Does the patient have a health problem that substantially limits life expectancy, "
                                  "or are they unwilling or unable to undergo curative lung surgery?", required=False),
    ),
)

AAA = Checklist(
    key="aaa",
    title="One-time abdominal aortic aneurysm ultrasound screening",
    source="USPSTF 2019 (Grade B), JAMA 2019;322(22):2211-2218",
    direct=("Under the USPSTF 2019 recommendation (one-time ultrasound screening for abdominal aortic aneurysm in "
            "men aged 65 to 75 years who have ever smoked), is this patient eligible for AAA screening?"),
    criteria=(
        Criterion("aaa_male", "Is the patient male?"),
        Criterion("aaa_age", "Is the patient between 65 and 75 years old, inclusive?"),
        Criterion("aaa_ever_smoked", "Has the patient ever smoked tobacco (current or former smoker)?"),
    ),
)

CHECKLISTS = {c.key: c for c in (LUNG, AAA)}
