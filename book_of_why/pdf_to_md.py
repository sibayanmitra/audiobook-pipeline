#!/usr/bin/env python3
"""Convert The Book of Why (Judea Pearl & Dana Mackenzie, calibre PDF) to Markdown.

Usage:
    python pdf_to_md.py [INPUT.pdf]

Outputs (next to this script):
    ./book_of_why.md     `# ` heading per chapter, `## ` per section heading
    ./chapters.json      [{num, title, start_page, end_page}, ...]

Layout: one column, LiberationSerif 14.4pt body, paragraphs indented (x~91, or ~154 inside
numbered lists), continuation at x=72 (135/126/109 inside lists). Section headings are
LiberationSans 16.6pt; chapter pages open with the number and title at 23.8pt and a drop-cap
first word at 26.6pt. Figure captions are 10.8pt (kept, minus photo credits).

Causal-diagram arrows inside sentences ("Fire -> Smoke -> Alarm") are tiny images, which
split a printed line into several text lines. Lines are rebuilt from spans grouped by
height, with the arrow images put back as characters; arrows and formulas are then turned
into words a speech engine can read ("P(Y | do(X))" -> "P of Y given do X").
An italic capital A is a variable, not the article, so it is written "Eigh" to be said "ay".
Not narrated: title, copyright, dedication, acknowledgments, notes, bibliography, index.
"""
import glob
import json
import re
import sys
from pathlib import Path

import pymupdf
from spellchecker import SpellChecker

HERE = Path(__file__).resolve().parent
OUT_MD = HERE / "book_of_why.md"
TERMINAL = tuple(".!?:;”\"')…—")
WRAP = "\x01"
SUB, SUB_END, SUP, SUP_END = "\x02", "\x03", "\x04", "\x05"
NARRATE = re.compile(r"^(Preface|INTRODUCTION .*|CHAPTER \d+ .*)$")
_dict = SpellChecker()

# inline images that are really text, by content digest (first 10 hex chars)
INLINE_IMAGES = {
    "3823bb8fed": " → ",
    "f4237808f0": " ← ",
    "6943698218": "√2",
    "25bfec29d8": "(i) X → Z → Y",
    "5c2e8584b2": "(ii) X → M → Y, with M → Z",
}
SMALL = {"a", "an", "and", "the", "of", "in", "on", "to", "for", "at", "by", "or", "but", "vs.",
         "over", "from", "with", "into"}
KEEP_CAPS = {"RCTS": "RCTs", "AI": "AI", "DNA": "DNA", "II": "II", "DR.": "Dr.", "MR.": "Mr.",
             "AD": "AD", "BC": "BC", "(CASE(S)": "(Case(s)", "CASE(S)": "Case(s)"}

FRACTIONS = {"1/2": "one half", "1/3": "one third", "2/3": "two thirds", "1/4": "one quarter",
             "3/4": "three quarters", "4/5": "four fifths", "5/12": "five twelfths",
             "1/5": "one fifth", "1/6": "one sixth", "1/10": "one tenth", "1/100": "one hundredth",
             "1/12": "one twelfth", "1/8": "one eighth", "1/116": "one in 116", "1/700": "one in 700"}

# hand-written readings for the formulas with sub- and superscripts that the text layer
# flattens (applied before the generic math rules); keys are regexes on the joined text
FORMULAS = [
    (r"\ba\x042\x05", "a squared"),
    (r"a = 1/\s*√2", "a equals one over the square root of 2"),
    (r"\bB = kP\b", "B equals k P"),
    (r"B–kP = 0", "B minus k P equals 0"),
    (r"\b(NDE|NIE|CDE|FAR)\b(?! requires)", lambda m: " ".join(m.group(1))),
    (r"\s*—\|\s*", " blunt arrow to "),                   # A —| B: A inhibits B
    (r"\bUS\b(?= ?(?:stands|=|\)|,|\())", "U sub S"),
    (r"\bU(EX|ED|Y)\b", r"U sub \1"),
    (r"\bf(S|ED|Y)\(", r"f sub \1 of ("),
    (r"\b(EX|ED)\((Alice|Bert|Caroline)\)", r"\1 of \2"),
    (r"\br([XYZ]{2})\b", r"r sub \1"),
    (r"\b([abc])([XYZ])\b", r"\1 \2"),
    (r"\bW([FI])\b", r"W sub \1"),
    (r"\b24/7\b", "twenty-four seven"),
    (r"≈", " is approximately "),
    (r"\bEigh (?=History\b)", "A "),
    (r"\bEigh/([A-Z])\b", r"Eigh over \1"),
    (r"percent/(?=\d)", "percent divided by "),
    (r"\bmg/dl\b", "milligrams per deciliter"),
    (r"\(5/12\) = \(1/2\)\(2/3\)", "× (5/12) = (1/2) × (2/3)"),
    (r"\b10–4 ×", "10 minus 4 ×"),
    (r"\b([NC] [DI] E)\((\d)\)", r"\1 of \2"),
]



def resolve_input(arg):
    if arg:
        return Path(arg)
    pdfs = sorted(glob.glob(str(HERE / "raw" / "*.pdf")))
    if not pdfs:
        sys.exit("No PDF found. Pass the path explicitly or place it in ./raw/")
    return Path(pdfs[0])


def build_sections(doc):
    toc = doc.get_toc()
    sections = []
    for i, (lvl, title, page) in enumerate(toc):
        if not NARRATE.match(title):
            continue
        nxt = toc[i + 1][2]
        m = re.match(r"CHAPTER (\d+) (.*)", title)
        if m:
            name = f"Chapter {m.group(1)}: {m.group(2)}"
        elif title.startswith("INTRODUCTION"):
            name = "Introduction: " + title.split(" ", 1)[1]
        else:
            name = title
        sections.append({"num": len(sections), "title": name, "start_page": page, "end_page": nxt - 1})
    return sections


def title_case(text):
    out = []
    for i, w in enumerate(text.split()):
        core = w.strip("()“”,.;:")
        if w.upper() in KEEP_CAPS:
            out.append(KEEP_CAPS[w.upper()])
        elif core.upper() in KEEP_CAPS and core:
            out.append(w.replace(core, KEEP_CAPS[core.upper()]))
        elif i and w.lower() in SMALL and not out[-1].endswith((":", "?", "—")):
            out.append(w.lower())
        else:
            out.append(re.sub(r"[^\W\d_][^\W\d_’']*",
                              lambda m: m.group(0)[0].upper() + m.group(0)[1:].lower(), w))
    return " ".join(out).replace("De Morgan", "De Morgan").replace("Ben Maimon", "ben Maimon")


def page_lines(page):
    """Rebuild printed lines from spans + inline images.
    -> [(x0, size, text, maxsize)], size = the size carrying most characters on the line.
    10.8pt spans sitting a few points below (above) a body line are subscripts (superscripts)
    of that line, wrapped in SUB/SUP markers; other 10.8pt spans are caption lines."""
    big, small = [], []   # (y_mid, x0, x1, size, text)
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                if not s["text"].strip():
                    continue
                text = s["text"]
                if "Italic" in s["font"]:
                    text = re.sub(r"(?<![\w’'])A(?![\w’'.])", "Eigh", text)  # variable A
                x0, y0, x1, y1 = s["bbox"]
                (big if s["size"] >= 12 else small).append(((y0 + y1) / 2, x0, x1, s["size"], text))
    for info in page.get_image_info(hashes=True):
        key = info["digest"].hex()[:10]
        if key in INLINE_IMAGES:
            x0, y0, x1, y1 = info["bbox"]
            big.append(((y0 + y1) / 2, x0, x1, 0, INLINE_IMAGES[key]))

    def group(items):
        rows = []
        for it in sorted(items, key=lambda t: (t[0], t[1])):
            if rows and abs(it[0] - rows[-1][0]) < 5.5:
                rows[-1][1].append(it)
            else:
                rows.append([it[0], [it]])
        return rows

    rows = group(big)
    loose = []
    for it in small:
        host, inline = None, False
        for r in rows:
            xs = [t[1] for t in r[1]] + [t[2] for t in r[1]]
            if not min(xs) - 2 <= it[1] <= max(xs) + 4:
                continue
            if abs(it[0] - r[0]) <= 3:            # small capitals on the same line
                host, inline = r, True
                break
            if 3 < abs(it[0] - r[0]) < 9:
                host = r
                break
        if host and inline:
            host[1].append((host[0], it[1], it[2], 0, it[4]))
        elif host and it[3] > 7:
            mark = (SUB, SUB_END) if it[0] > host[0] else (SUP, SUP_END)
            host[1].append((host[0], it[1], it[2], 0, mark[0] + it[4].strip() + mark[1]))
        else:
            loose.append(it)
    rows += group(loose)

    out = []
    for _, row in sorted(rows, key=lambda r: r[0]):
        row.sort(key=lambda t: t[1])
        text, last_x1 = "", None
        weight = {}
        for _, x0, x1, size, t in row:
            glued = t.startswith((SUB, SUP))
            if (last_x1 is not None and not glued and x0 - last_x1 > 1.0
                    and not text.endswith(" ") and not t.startswith(" ")):
                text += " "
            text += t
            last_x1 = x1
            if size:
                weight[round(size, 1)] = weight.get(round(size, 1), 0) + len(t)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            size = max(weight, key=weight.get) if weight else 14.4
            out.append((row[1][0][1] if False else min(t[1] for t in row), size, text,
                        max(weight) if weight else 14.4))
    return out


def new_para(x0, text):
    return (88 <= x0 <= 95 or 150 <= x0 <= 158 or text.startswith(("•", "—", "GAME "))
            or (115 <= x0 <= 122 and re.match(r"\d+\. ", text))
            or (85 <= x0 <= 89 and re.match(r"\([a-d]\) ", text)))


def section_items(doc, sec):
    items, cur, cap, head = [], "", "", ""
    prev_x = 0

    def flush():
        nonlocal cur, cap, head
        if head:
            items.append(("head", head))
        if cap:
            items.append(("para", cap))
        if cur:
            items.append(("para", cur))
        cur, cap, head = "", "", ""

    def join(a, b):
        if not a:
            return b
        if a.endswith("-") and b[:1].islower():
            return a[:-1] + WRAP + b
        return f"{a} {b}"

    for pno in range(sec["start_page"], sec["end_page"] + 1):
        lines = page_lines(doc[pno - 1])
        start = pno == sec["start_page"]
        for x0, size, text, maxsize in lines:
            if size > 20 and maxsize < 25:                       # chapter number / title
                continue
            if start and 16 < size < 17 and not items and not cur:  # PREFACE / INTRODUCTION title
                continue
            if 16 < size < 17:                                   # section heading (may wrap)
                if cur or cap:
                    flush()
                head = f"{head} {text}".strip()
                continue
            if 10 < size < 11:                                   # figure caption
                if cur and not cap:
                    items.append(("para", cur))
                    cur = ""
                if head:
                    items.append(("head", head))
                    head = ""
                cap = join(cap, text)
                continue
            if cap:                                              # caption ended
                items.append(("para", cap))
                cap = ""
            if head:
                items.append(("head", head))
                head = ""
            if maxsize > 25:                                     # drop-cap first word: IN -> In
                text = re.sub(r"^([“‘\"]?)([A-Z])([A-Z’']+)\b",
                              lambda m: m.group(1) + m.group(2) + m.group(3).lower(), text)
                flush()
                cur = text
                continue
            quote_cont = 88 <= x0 <= 95 and 88 <= prev_x <= 95 and not cur.endswith(TERMINAL)
            prev_x = x0
            if (new_para(x0, text) and not quote_cont) or not cur:
                if cur:
                    items.append(("para", cur))
                cur = text
            else:
                cur = join(cur, text)
    flush()
    return [(k, title_case(t)) if k == "head" else (k, t) for k, t in items]


def resolve_wraps(text):
    def pick(m):
        left, right = m.group(1), m.group(2)
        lw, rw = re.sub(r"\W", "", left).lower(), re.sub(r"\W", "", right).lower()
        if lw + rw in _dict or lw not in _dict or rw not in _dict:
            return left + right
        return f"{left}-{right}"
    return re.sub(r"([\w’']+)" + WRAP + r"([\w’']+)", pick, text)


def speak_parens(text):
    """P(L | do(D)) -> 'P of L given do D'; drops the brackets that belong to P( and do(."""
    out, stack, i = [], [], 0
    while i < len(text):
        m = re.match(r"(?<![\w])(P|do)\(", text[i:]) if (i == 0 or not text[i - 1].isalnum()) else None
        if m:
            out.append("P of " if m.group(1) == "P" else "do ")
            stack.append(True)
            i += m.end()
            continue
        c = text[i]
        if c == "(":
            stack.append(False)
            out.append(c)
        elif c == ")" and stack:
            if not stack.pop():
                out.append(c)
        else:
            out.append(c)
        i += 1
    return "".join(out)


def speak_math(text):
    for pat, rep in FORMULAS:
        text = re.sub(pat, rep, text)
    text = text.replace(SUB_END + SUB, " ")
    text = re.sub(r"\)\s+(?=[∑Σ])", ") × ", text)
    text = re.sub(SUP + r"\*" + SUP_END, " star", text)
    text = re.sub(r"[∑Σ]\s?" + SUB + r"([a-z])" + SUB_END, r"the sum over \1 of", text)
    text = re.sub(SUP + "2" + SUP_END, " squared", text)
    text = re.sub(SUP + r"([^" + SUP_END + r"]*)" + SUP_END, r" to the power of \1", text)
    text = re.sub(SUB + r"([^" + SUB_END + r"]*)" + SUB_END,
                  lambda m: f" sub {m.group(1)}" + ("," if " " in m.group(1) else ""), text)
    text = re.sub(r",?\s*\((u|Alice|Bert|Caroline)\)", r" of \1", text)
    text = re.sub(r"\)\s+(?=P\()", ") × ", text)                  # P(S | T) P(T): a product
    text = re.sub(r"(?<=[\w)])/(?=P\()", " divided by ", text)
    text = re.sub(r"\)/\(", ") over (", text)
    text = re.sub(r"(?<=\w)/(?=r sub)", " over ", text)
    text = re.sub(r"\b([A-Za-z])/([A-Za-z])\b", r"\1 over \2", text)
    text = re.sub(r"\bP\(([^)]*?) AND ", r"P(\1 and ", text)
    text = re.sub(r"(?<=[\w)])\s?–\s?(?=P\(|\(|[A-Z]\b|S sub)", " minus ", text)
    text = re.sub(r"(?<![\w)])–(\d)", r"negative \1", text)
    if re.search(r"\b(P|do)\(", text):
        text = speak_parens(text)
    text = re.sub(r" ?\| ?", " given ", text)
    text = re.sub(r"(?<=[A-Za-z])‴", " triple prime", text)
    text = re.sub(r"(?<=[A-Za-z])″", " double prime", text)
    text = re.sub(r"(?<=[A-Za-z])′", " prime", text)
    text = re.sub(r"(?<=\d)″", " inches", text)
    text = re.sub(r"[∑Σ]\s?([a-z])\b", r"the sum over \1 of", text)
    text = re.sub(r"√2", "the square root of 2", text)
    text = re.sub(r"\b(\d+/\d+)\b", lambda m: FRACTIONS.get(m.group(1), m.group(1).replace("/", " over ")), text)
    text = re.sub(r"\(\s*(one half|one third|two thirds|four fifths|five twelfths)\s*\)", r"\1", text)
    text = re.sub(r"\b([A-Za-z])(\d)\(([^()]+)\)", r"\1 sub \2 of \3", text)   # S1(Bert)
    text = re.sub(r"(?<=\s)(\d)\((u)\)", r"\1 of \2", text)                        # =0(u)
    text = re.sub(r"\b([A-Za-z])(\d)\b(?!\.\d)", r"\1 sub \2", text)              # W0, c1, U2
    text = re.sub(r"(?<![\w])~(?=[A-Z]\b)", "not ", text)                            # ~D
    text = re.sub(r"\b([A-Z])\*", r"\1 star", text)                                 # L*
    text = re.sub(r"\s=\s", " equals ", text)
    text = re.sub(r"\s>\s", " is greater than ", text)
    text = re.sub(r"\s<\s", " is less than ", text)
    text = re.sub(r"\s≠\s", " is not equal to ", text)
    text = re.sub(r"\s×\s(?:×\s)?", " times ", text)
    text = re.sub(r"(?<=[\w)])\s\+\s(?=[\w($])", " plus ", text)
    text = re.sub(r"(?<=\w) – (?=\w)", " minus ", text) if re.search(r"equals|given", text) else text
    text = re.sub(r"\s→\s", " arrow to ", text)
    text = re.sub(r"\s←\s", " arrow from ", text)
    text = re.sub(r"\bA(?= arrow (?:to|from) )|(?<=arrow to )A\b|(?<=arrow from )A\b", "Eigh", text)
    text = re.sub(r"\s+,", ",", text)
    return text


FIXUPS = [
    (r"\(Source: [^)]*(?:\([^)]*\)[^)]*)*\)", ""),            # photo/figure credits
    (r"^•\s*", ""),
    (r"^GAME (\d+)\.", r"Game \1."),
    (r"^—\s*([A-ZÀ-Ý][A-ZÀ-Ý .()0-9–:’'-]+)$", lambda m: "— " + title_case(m.group(1))),
    (r"\b(FIGURE|TABLE) ", lambda m: m.group(1).title() + " "),
    (r"\s+(\(\d{1,2}\.\d\))$", r" \1"),
    (r"(?<=\s)\((\d{1,2}\.\d)\)(?=\s|$)", r"(Equation \1)"),
    (r"\s*…\s*", "… "),
    (r"(\w)\[(\w+)\]", r"\1\2"),                      # know[s] -> knows, 6.9[b] -> 6.9b
    (r"(\d)\.(\d)([a-d])\b", r"\1.\2 \3"),             # Figure 6.9b -> 6.9 b
    (r"\[([^\]]+)\]", r"(\1)"),                         # editorial [insertions]
    (r"\b(\d+(?:\.\d+)?)(foot|point)\b", r"\1-\2"),
]


def finish(text):
    text = resolve_wraps(text).replace(WRAP, "")
    for pat, rep in FIXUPS:
        text = re.sub(pat, rep, text)
    text = speak_math(text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text if text.endswith(TERMINAL + (",",)) else text + "."


def main():
    src = resolve_input(sys.argv[1] if len(sys.argv) > 1 else None)
    doc = pymupdf.open(src)
    sections = build_sections(doc)
    print(f"{src.name}: {len(doc)} pages, {len(sections)} sections")
    md = []
    for sec in sections:
        md.append(f"# {sec['title']}\n")
        for kind, text in section_items(doc, sec):
            md.append(f"## {text}\n" if kind == "head" else finish(text) + "\n")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    (HERE / "chapters.json").write_text(json.dumps(sections, indent=2), encoding="utf-8")
    print(f"wrote {OUT_MD.name} ({sum(len(m.split()) for m in md):,} words) and chapters.json")


if __name__ == "__main__":
    main()
