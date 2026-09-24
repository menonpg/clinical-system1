"""Build a balanced TREC Clinical Trials 2021 eligible-vs-excluded sample with trial criteria from ClinicalTrials.gov."""
import json, random, re, time, urllib.request, xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "data" / "trec_ct"
N_PER_CLASS = 300
random.seed(13)

topics = {t.get("number"): t.text.strip() for t in ET.parse(D / "topics2021.xml").getroot()}
by_label = defaultdict(list)
for line in open(D / "qrels2021.txt"):
    tid, _, nct, rel = line.split()
    if rel in ("1", "2"):
        by_label[rel].append((tid, nct))

def stratified(pairs, n):
    per = defaultdict(list)
    for p in pairs: per[p[0]].append(p)
    for v in per.values(): random.shuffle(v)
    out, keys = [], sorted(per, key=int)
    while len(out) < n:
        for k in keys:
            if per[k] and len(out) < n: out.append(per[k].pop())
    return out

sample = [(t, n, 1) for t, n in stratified(by_label["2"], N_PER_CLASS)] + \
         [(t, n, 0) for t, n in stratified(by_label["1"], N_PER_CLASS)]
cache = D / "trials"; cache.mkdir(exist_ok=True)

def fetch(nct):
    f = cache / f"{nct}.json"
    if f.exists(): return json.loads(f.read_text())
    url = (f"https://clinicaltrials.gov/api/v2/studies/{nct}?fields=protocolSection.identificationModule.briefTitle,"
           "protocolSection.eligibilityModule,protocolSection.conditionsModule")
    for _ in range(3):
        try:
            j = json.load(urllib.request.urlopen(url, timeout=30)); break
        except Exception as e:
            err = e; time.sleep(2)
    else:
        raise err
    f.write_text(json.dumps(j)); time.sleep(0.1)
    return j

def split_criteria(text):
    m = re.split(r"(?im)^\s*exclusion criteria\s*:?", text, maxsplit=1)
    inc = re.sub(r"(?im)^\s*inclusion criteria\s*:?", "", m[0]).strip()
    exc = m[1].strip() if len(m) > 1 else ""
    return inc, exc

rows = []
for tid, nct, y in sample:
    try:
        ps = fetch(nct)["protocolSection"]
    except Exception as e:
        print("skip", nct, e); continue
    el = ps.get("eligibilityModule", {})
    inc, exc = split_criteria(el.get("eligibilityCriteria", ""))
    rows.append(dict(topic=tid, nct=nct, label=y, patient=topics[tid],
                     title=ps["identificationModule"].get("briefTitle", ""),
                     conditions=ps.get("conditionsModule", {}).get("conditions", []),
                     sex=el.get("sex", "ALL"), min_age=el.get("minimumAge", ""), max_age=el.get("maximumAge", ""),
                     inclusion=inc, exclusion=exc))
(D / "sample.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print(len(rows), "pairs;", sum(r["label"] for r in rows), "eligible")
