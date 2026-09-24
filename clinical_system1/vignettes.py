"""Synthetic primary-care vignettes with exact ground truth for USPSTF screening checklists.

Every vignette is rendered from a structured seed, so each criterion label is known exactly.
Surface forms vary on purpose: pack-years come as packs/day x years, cigarettes/day or an explicit
pack-year figure; quitting comes as a calendar year or "N years ago". Distractors include household
smoking, family history and benign comorbidities. Boundary cases are oversampled.

Templates come in two disjoint families ("A" and "B") so we can fine-tune on one and test on the other.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

NOTE_YEAR = 2026

LIMITING = [
    "metastatic pancreatic adenocarcinoma; enrolled in home hospice",
    "NYHA class IV heart failure with an estimated prognosis of under 6 months",
    "end-stage COPD on 4 L home oxygen, not a surgical candidate per pulmonology",
    "advanced Alzheimer dementia, bed-bound, comfort-focused care per goals-of-care discussion",
    "ESRD, has declined dialysis; palliative care following",
]
DECLINES_SURGERY = [
    "States clearly that she would never agree to any lung surgery, even if cancer were found.",
    "He is adamant he would refuse any operation or biopsy regardless of findings.",
    "Patient reports she would not undergo surgery under any circumstances.",
]
BENIGN = ["hypertension", "type 2 diabetes (A1c 6.9%)", "hyperlipidemia", "osteoarthritis of the knees",
          "GERD", "hypothyroidism on levothyroxine", "BPH", "mild intermittent asthma", "depression, stable on sertraline",
          "obesity (BMI 33)", "gout"]
DISTRACTORS = ["Spouse smokes indoors.", "Father died of lung cancer at 71.", "Works as a bartender; notes heavy secondhand smoke exposure in the past.",
               "Mother had COPD.", "Brother recently diagnosed with an abdominal aortic aneurysm.", "Drinks 2-3 beers on weekends.",
               "Walks 30 minutes daily.", "Retired machinist."]
REASONS = ["annual wellness visit", "follow-up of blood pressure", "medication refill", "knee pain", "establishing care",
           "diabetes follow-up", "routine physical"]


@dataclass
class Seed:
    id: str
    age: int
    sex: str               # "M" | "F"
    status: str            # "never" | "current" | "former"
    ppd: float             # packs per day while smoking
    years: int             # years smoked
    quit_ago: int | None   # years since quitting (former only)
    limiting: str | None   # life-limiting condition text
    declines: bool         # unwilling to undergo curative surgery
    family: str            # template family "A" | "B"
    pack_form: str = ""    # how smoking is phrased
    quit_form: str = ""

    # ---- ground truth --------------------------------------------------
    @property
    def pack_years(self) -> float:
        return 0.0 if self.status == "never" else round(self.ppd * self.years, 1)

    def labels(self) -> dict[str, bool]:
        lcs = {
            "lcs_age": 50 <= self.age <= 80,
            "lcs_pack_years": self.pack_years >= 20,
            "lcs_recency": self.status == "current" or (self.status == "former" and self.quit_ago is not None and self.quit_ago <= 15),
            "lcs_limiting": bool(self.limiting) or self.declines,
        }
        lcs["lcs_eligible"] = lcs["lcs_age"] and lcs["lcs_pack_years"] and lcs["lcs_recency"] and not lcs["lcs_limiting"]
        aaa = {
            "aaa_male": self.sex == "M",
            "aaa_age": 65 <= self.age <= 75,
            "aaa_ever_smoked": self.status != "never",
        }
        aaa["aaa_eligible"] = all(aaa.values())
        return {**lcs, **aaa}


def _pick_age(r: random.Random) -> int:
    u = r.random()
    if u < 0.35:  # boundary oversampling for both checklists
        return r.choice([48, 49, 50, 51, 63, 64, 65, 66, 74, 75, 76, 79, 80, 81, 82])
    return r.randint(40, 90)


def sample_seed(r: random.Random, i: int, family: str) -> Seed:
    age = _pick_age(r)
    status = r.choices(["never", "current", "former"], [0.2, 0.4, 0.4])[0]
    ppd, years, quit_ago = 0.0, 0, None
    if status != "never":
        start = r.randint(14, 25)
        max_years = max(1, age - start)
        if r.random() < 0.35:  # pack-year boundary
            target = r.choice([15, 18, 19, 20, 21, 22, 25])
            ppd = r.choice([0.5, 1.0, 1.0, 2.0])
            years = max(1, min(max_years, round(target / ppd)))
        else:
            ppd = r.choice([0.25, 0.5, 0.5, 1.0, 1.0, 1.5, 2.0])
            years = r.randint(1, max_years)
        if status == "former":
            quit_ago = r.choice([2, 5, 8, 12, 14, 15, 16, 17, 20, 25, 30]) if r.random() < 0.6 else r.randint(1, 35)
            quit_ago = min(quit_ago, max(1, age - start - years)) if age - start - years > 0 else 1
            years = min(years, max(1, age - start - quit_ago))
    lim = r.choice(LIMITING) if r.random() < 0.12 else None
    declines = (not lim) and r.random() < 0.06
    return Seed(id=f"{family}{i:04d}", age=age, sex=r.choice("MF"), status=status, ppd=ppd, years=years,
                quit_ago=quit_ago, limiting=lim, declines=declines, family=family)


# ---------------------------------------------------------------------------- rendering
def _smoking_phrase(s: Seed, r: random.Random) -> str:
    if s.status == "never":
        s.pack_form = "never"
        return r.choice(["Never smoker.", "Denies ever using tobacco.", "Lifelong nonsmoker.", "Tobacco: never."])
    forms = ["ppd", "cigs", "packyears"]
    f = r.choice(forms)
    s.pack_form = f
    if f == "ppd":
        amt = {0.25: "a quarter pack", 0.5: "half a pack", 1.0: "1 pack", 1.5: "1.5 packs", 2.0: "2 packs"}[s.ppd]
        core = f"{amt} per day for {s.years} years"
    elif f == "cigs":
        core = f"about {int(s.ppd * 20)} cigarettes a day for {s.years} years"
    else:
        py = s.pack_years
        core = f"a {py:g} pack-year history"
    if s.status == "current":
        return r.choice([f"Current smoker, {core}.", f"Smokes {core}; still smoking.", f"Active tobacco use: {core}."]) \
            if f != "packyears" else r.choice([f"Current smoker with {core}.", f"Still smokes; {core}."])
    qy = NOTE_YEAR - s.quit_ago
    s.quit_form = r.choice(["ago", "year"])
    quit = f"quit {s.quit_ago} years ago" if s.quit_form == "ago" else f"quit in {qy}"
    if f == "packyears":
        return r.choice([f"Former smoker with {core}, {quit}.", f"Ex-smoker ({core}); {quit}."])
    return r.choice([f"Former smoker: smoked {core}, {quit}.", f"Previously smoked {core} and {quit}."])


def _sex_words(s: Seed):
    return ("man", "male", "He", "his") if s.sex == "M" else ("woman", "female", "She", "her")


def render(s: Seed, r: random.Random) -> str:
    noun, adj, Pro, pos = _sex_words(s)
    pmh = r.sample([b for b in BENIGN if not (b == "BPH" and s.sex == "F")], r.randint(1, 3))
    if s.limiting:
        pmh.insert(r.randint(0, len(pmh)), s.limiting)
    smoke = _smoking_phrase(s, r)
    extra = " ".join(r.sample(DISTRACTORS, r.randint(0, 2)))
    decl = r.choice(DECLINES_SURGERY) if s.declines else ""
    if s.declines and s.sex == "M":
        decl = decl.replace("she ", "he ").replace("She ", "He ")
    elif s.declines:
        decl = decl.replace("He ", "She ").replace(" he ", " she ")
    reason = r.choice(REASONS)
    if s.family == "A":
        t = r.randint(0, 1)
        if t == 0:
            txt = (f"{s.age}-year-old {noun} presenting for {reason}. PMH: {', '.join(pmh)}. "
                   f"Social history: {smoke} {extra} {decl}")
        else:
            txt = (f"CC: {reason}\nHPI: {s.age} y/o {adj}.\nPast medical history:\n" + "\n".join(f"- {p}" for p in pmh) +
                   f"\nSocial: {smoke} {extra}\n{decl}")
    else:
        t = r.randint(0, 1)
        if t == 0:
            txt = (f"Patient seen today ({NOTE_YEAR}) for {reason}. {Pro} is {s.age} years old. "
                   f"{Pro} has a history of {', '.join(pmh[:-1]) + ' and ' + pmh[-1] if len(pmh) > 1 else pmh[0]}. "
                   f"Regarding tobacco: {smoke} {extra} {decl}")
        else:
            txt = (f"ASSESSMENT NOTE {NOTE_YEAR}\nDemographics: age {s.age}, sex {adj}\nProblem list: {'; '.join(pmh)}\n"
                   f"Habits: {smoke}\nOther: {extra or 'none'} {decl}")
    if "quit in" in smoke and s.family == "A":
        txt = f"Visit date: {NOTE_YEAR}. " + txt
    return " ".join(txt.split(" ")).strip()


def build(n: int, family: str, seed: int, target: str | None = None, pos_frac: float = 0.4) -> list[dict]:
    """Render n vignettes. With target (e.g. "lcs_eligible"), rejection-sample to pos_frac positives."""
    r = random.Random(seed)
    need_pos = round(n * pos_frac) if target else None
    out, npos, i = [], 0, 0
    while len(out) < n:
        s = sample_seed(r, i, family)
        i += 1
        lab = s.labels()
        if target:
            if lab[target] and npos >= need_pos:
                continue
            if not lab[target] and len(out) - npos >= n - need_pos:
                continue
            npos += lab[target]
        text = render(s, r)
        out.append({"id": s.id, "text": text, "labels": lab, "seed": asdict(s), "pack_years": s.pack_years})
    return out


if __name__ == "__main__":
    import json, sys
    for v in build(int(sys.argv[1]) if len(sys.argv) > 1 else 5, "B", 1):
        print(json.dumps(v["labels"]), "\n", v["text"], "\n")
