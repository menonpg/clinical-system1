"""Deterministic fact augmentation for smoking arithmetic (a pre-processing hook, like Kev's KEV_DATE_FACTS).

Decision models are weak at arithmetic over the note: packs/day x years, cigarettes -> packs, and
"quit in 2010" against the visit year. This hook finds those spans with generic regular expressions
and appends plain sentences stating the computed values:

    [Computed facts: about 40 pack-years (1 packs/day x 40 years). Quit about 16 years before the visit (2010 -> 2026).]

The model still makes every judgement; the hook only supplies arithmetic. When a pattern does not
match, nothing is added, so a miss degrades to the un-augmented note.
"""
from __future__ import annotations

import re

WORDNUM = {"a": 1, "one": 1, "two": 2, "three": 3, "half a": 0.5, "a half": 0.5, "a quarter": 0.25,
           "quarter": 0.25, "half": 0.5, "1.5": 1.5, "one and a half": 1.5, "a pack and a half": 1.5}
NUM = r"(?:\d+(?:\.\d+)?|a quarter|half a|a half|one and a half|a|one|two|three)"


def _num(s: str) -> float:
    s = s.strip().lower()
    return WORDNUM[s] if s in WORDNUM else float(s)


def visit_year(text: str, default: int = 2026) -> int:
    m = re.search(r"(?:visit|seen|date of service|note|encounter|annual|wellness|physical|f/u|follow-up)[^.\n]{0,25}?\b(20[0-4]\d)\b", text, re.I) \
        or re.search(r"\((20[0-4]\d)\)", text) or re.search(r"^\s*(20[0-4]\d)\b", text)
    return int(m.group(1)) if m else default


def age_of(text: str) -> int | None:
    m = re.search(r"\b(\d{2})[- ]?(?:-year-old|year-old|year old|years old|yo|y/o|y\.o\.|yrs? old)\b", text, re.I) \
        or re.search(r"\bage (\d{2})\b", text, re.I) or re.search(r"\b(\d{2}) ?(?:M|F)\b", text)
    return int(m.group(1)) if m else None


def packs_per_day(text: str) -> float | None:
    t = text.lower()
    if re.search(r"a pack and a half|one and a half packs?", t):
        return 1.5
    m = re.search(r"(\d+)\s*(?:cigarettes|cigs)\s*(?:a|per|/)\s*day", t) or re.search(r"(\d+)\s*cigarettes daily", t)
    if m:
        return int(m.group(1)) / 20
    m = re.search(r"one pack every two days|a pack every other day|pack every 2 days", t)
    if m:
        return 0.5
    m = re.search(rf"~?({NUM})\s*(?:ppd|packs?\s*(?:a|per|/)\s*day|packs?/day|packs? daily|pack a day|pack-a-day)", t)
    if m:
        return _num(m.group(1))
    return None


def years_smoked(text: str, age: int | None, vy: int) -> float | None:
    t = text.lower()
    m = re.search(r"(?:for|x)\s*(?:about |approximately |roughly |nearly |~|the last |the past )?(\d+)\s*(?:years|yrs|yr|y)\b", t)
    if m:
        return float(m.group(1))
    m = re.search(r"from (?:age )?(\d{2}) (?:to|until) (?:age )?(\d{2})\b", t)
    if m:
        return float(int(m.group(2)) - int(m.group(1)))
    m = re.search(r"(?:since (?:he was |she was |age )?|started at (?:age )?)(\d{2})\b", t)
    if m and age:
        return float(age - int(m.group(1)))
    return None


def quit_years_ago(text: str, age: int | None, vy: int) -> tuple[float, str] | None:
    t = text.lower()
    m = re.search(r"quit[^.;]{0,40}?\b((?:19|20)\d\d)\b", t) or re.search(r"stopped[^.;]{0,20}?\b((?:19|20)\d\d)\b", t)
    if m:
        y = int(m.group(1))
        return vy - y, f"{y} -> {vy}"
    if re.search(r"quit (?:smoking )?last year", t):
        return 1.0, "last year"
    m = re.search(r"quit (?:at|when (?:he|she) was) (?:age )?(\d{2})\b", t) or re.search(r"(?:to|until) (?:age )?(\d{2}),? then quit", t)
    if m and age:
        return float(age - int(m.group(1))), f"at age {m.group(1)}, now {age}"
    return None


def augment(text: str) -> str:
    if re.search(r"never smok|denies (?:ever )?(?:using )?tobacco|non-?smoker|tobacco: never|never used tobacco", text, re.I):
        return text
    vy, age = visit_year(text), age_of(text)
    facts = []
    ppd = packs_per_day(text)
    stated = re.search(r"(\d+(?:\.\d+)?)[ -]?(?:pack[- ]years?|pk[- ]?yrs?|pack-year)", text, re.I)
    if ppd is not None:
        yrs = years_smoked(text, age, vy)
        if yrs is not None:
            facts.append(f"about {ppd * yrs:g} pack-years ({ppd:g} packs/day x {yrs:g} years)")
        else:
            facts.append(f"{ppd:g} packs per day")
    elif stated:
        facts.append(f"{float(stated.group(1)):g} pack-years as stated")
    q = quit_years_ago(text, age, vy)
    if q is not None:
        facts.append(f"quit about {q[0]:g} years before the visit ({q[1]})")
    if age is not None:
        facts.append(f"age {age}")
    return text if not facts else f"{text}\n[Computed facts: {'; '.join(facts)}.]"
