"""Assemble docs/*.html from docs/src/*.html bodies with a shared shell and navigation.

usage: python docs/build.py
"""
from pathlib import Path

D = Path(__file__).parent
PAGES = [  # (file, nav label, <title>, group)
    ("index.html", "Overview", "Clinical System One", "Project"),
    ("app.html", "App & demo", "Screening app", "Project"),
    ("explorer.html", "Screening Explorer", None, "Project"),
    ("landscape.html", "The field, Nov 2026", "Decision models, November 2026", "Upstream models"),
    ("laya.html", "Laya", "Laya, studied", "Upstream models"),
    ("kev.html", "Kev", "Kev, studied", "Upstream models"),
    ("compare.html", "Laya vs Kev", "Laya vs Kev", "Upstream models"),
    ("clinical.html", "Study design", "Checklist study design", "Clinical study"),
    ("results.html", "Results", "Checklist study results", "Clinical study"),
    ("reproduce.html", "Reproduce", "Reproduce the study", "Clinical study"),
]


def nav(current):
    out, grp = ['<a class="brand" href="index.html">Clinical<br>System One</a>'], None
    for f, label, _, g in PAGES:
        if g != grp:
            out.append(f'<div class="grp">{g}</div>')
            grp = g
        cur = ' aria-current="page"' if f == current else ""
        out.append(f'<a href="{f}"{cur}>{label}</a>')
    return "\n".join(out)


for f, _, title, _ in PAGES:
    src = D / "src" / f
    if title is None or not src.exists():
        continue
    body = src.read_text()
    (D / f).write_text(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{title}</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<div class="shell">
<nav class="side" aria-label="Documentation">
{nav(f)}
</nav>
<main>
{body}
<footer>Research prototype, not a medical device. Code: Apache-2.0. Upstream: Laya (NandhaKishorM/laya), Kev (jaredpalmer/kev), both Apache-2.0.</footer>
</main>
</div>
</body>
</html>
""")
    print("wrote", f)
