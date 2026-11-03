#!/usr/bin/env python3
"""Convert The Hard Thing About Hard Things (Ben Horowitz, calibre PDF) to Markdown.

Usage:
    python pdf_to_md.py [INPUT.pdf]

Outputs (next to this script):
    ./hard_thing.md     `# ` heading per chapter, `## ` for essays and their subheadings
    ./chapters.json     [{num, label, title, start_page, end_page}, ...]

Layout: one column, no running headers or page numbers, one body font size (14.4pt).
Paragraphs: first line indented (x~86), continuation at x~72. Bullets start at x~107 and
continue at x~101. Epigraphs (rap lyrics) are centred plain lines followed by a bold
"—ARTIST, “SONG”" line. Headings are bold ALL-CAPS lines at any size; chapter titles
(and their "— CHAPTER N —" label) are dropped because the audiobook announces the chapter.
Level-2 TOC entries (the essays in chapters 4-8) begin on a new page and become `## `.
Not narrated: dedication, acknowledgments, about the author, credits, copyright.
"""
import glob
import json
import re
import sys
from pathlib import Path

import pymupdf
from spellchecker import SpellChecker

HERE = Path(__file__).resolve().parent
OUT_MD = HERE / "hard_thing.md"

NARRATE = re.compile(r"^(Introduction|Chapter \d+:.*|Appendix:.*)$")
INDENT = (82, 92)       # first line of a paragraph
BULLET = (104, 111)     # first line of a bullet item
CENTRED = 115           # epigraph lines / headings start further right than any body line
TERMINAL = tuple(".!?:;”\"')’…—")
WRAP = "\x01"
ACRONYMS = {                                    # kept upper-case when headings are title-cased
    "CEO": "CEO", "CEOS": "CEOs", "DMX": "DMX", "VP": "VP", "HR": "HR", "IPO": "IPO",
    "CFO": "CFO", "CTO": "CTO", "NBA": "NBA", "II": "II", "RZA": "RZA", "GZA": "GZA",
    "KRS-ONE": "KRS-One", "QA": "QA", "Z": "Z", "MC": "MC", "OK": "OK", "VS.": "vs.",
    "LL": "LL", "EPMD": "EPMD", "A$AP": "A$AP", "UGK": "UGK", "MOP": "M.O.P."}
SMALL = {"a", "an", "and", "the", "of", "in", "on", "to", "for", "at", "by", "or", "but"}
_dict = SpellChecker()

FIXUPS = [
    (r"\s*…\s*", "… "),
]


def resolve_input(arg):
    if arg:
        return Path(arg)
    pdfs = sorted(glob.glob(str(HERE / "raw" / "*.pdf")))
    if not pdfs:
        sys.exit("No PDF found. Pass the path explicitly or place it in ./raw/")
    return Path(pdfs[0])


def tidy_toc(title):
    return re.sub(r"\s+", " ", title).strip()


def build_sections(doc):
    toc = [(lvl, tidy_toc(t), p) for lvl, t, p in doc.get_toc()]
    sections = []
    for i, (lvl, title, page) in enumerate(toc):
        if lvl != 1 or not NARRATE.match(title):
            continue
        nxt = next((p for l, _, p in toc[i + 1:] if l == 1), len(doc) + 1)
        subs = [(t, p) for l, t, p in toc[i + 1:] if l == 2 and p < nxt]
        m = re.match(r"Chapter (\d+): (.*)", title)
        sections.append({"num": int(m.group(1)) if m else (0 if title == "Introduction" else 10),
                         "label": title, "title": m.group(2) if m else title,
                         "start_page": page, "end_page": nxt - 1, "essays": subs})
    for s in sections:
        if s["label"].startswith("Chapter"):
            s["title"] = f"Chapter {s['num']}: {s['title']}"
    return sections


def is_bold(spans):
    return all("Bold" in s["font"] for s in spans if s["text"].strip())


def caps(text):
    letters = re.sub(r"[^A-Za-z]", "", text)
    return len(letters) >= 2 and letters.isupper()


def title_case(text):
    """'SOME STUFF THAT MAY OR MAY NOT HELP' -> 'Some Stuff That May or May Not Help'."""
    out = []
    for i, w in enumerate(text.split()):
        core = w.strip("“”\"’'(),:;!?.—")
        if core.upper() in ACRONYMS:
            out.append(w.replace(core, ACRONYMS[core.upper()]))
        elif i and core.lower() in SMALL:
            out.append(w.lower())
        else:
            out.append(re.sub(r"[A-Za-z][A-Za-z’']*", lambda m: m.group(0)[0].upper() + m.group(0)[1:].lower(), w))
    return " ".join(out)


def page_lines(page):
    """-> [(x0, kind, text, size)], kind = label | head | attrib | epi | bq | body."""
    out = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            spans = l["spans"]
            text = "".join(s["text"] for s in spans).strip()
            if not text:
                continue
            x0, bold = l["bbox"][0], is_bold(spans)
            size = max(s["size"] for s in spans if s["text"].strip())
            if re.fullmatch(r"—\s*CHAPTER \d+\s*—|APPENDIX", text):
                kind = "label"
            elif bold and text.startswith("—"):
                kind = "attrib"
            elif bold and caps(text):
                kind = "head"
            elif bold and size < 15 and x0 < 80:
                kind = "bq"                       # bold question lines in the appendix
            elif not bold and x0 > CENTRED:
                kind = "epi"
            else:
                kind = "body"
            out.append((x0, kind, text, size))
    return out


def section_items(doc, sec):
    items, cur, epi = [], "", []
    essays = {p: t for t, p in sec["essays"]}

    def flush():
        nonlocal cur
        if cur:
            items.append(("para", cur))
        cur = ""

    def flush_epi():
        nonlocal epi
        if epi:
            flush()
            joined = ""
            for line in epi:
                joined = line if not joined else (f"{joined} {line}" if joined.endswith(TERMINAL + (",",))
                                                   else f"{joined}, {line}")
            items.append(("para", joined))
            epi = []

    for pno in range(sec["start_page"], sec["end_page"] + 1):
        lines = page_lines(doc[pno - 1])
        title_zone = pno == sec["start_page"] or pno in essays
        if title_zone:
            # an epigraph's widest lyric line can reach the left margin: every plain line
            # before the first "—ARTIST" attribution on a title page belongs to the epigraph
            kinds = [k for _, k, _, _ in lines]
            if "attrib" in kinds[:14]:
                cut = kinds.index("attrib")
                lines = [(x, "epi" if k == "body" and i < cut else k, t, sz)
                         for i, (x, k, t, sz) in enumerate(lines)]
        if pno in essays:
            flush_epi()
            flush()
            items.append(("head", essays[pno]))
        for x0, kind, text, size in lines:
            if kind == "label":
                continue
            if kind == "head" and (size > 15 and title_zone):
                continue                              # chapter / essay title (announced separately)
            if kind != "head":
                title_zone = False
            if kind == "epi":
                flush()
                epi.append(text)
                continue
            if kind == "attrib":
                flush_epi()
                flush()
                items.append(("para", "— " + title_case(text.lstrip("— "))))
                continue
            flush_epi()
            if kind == "head":
                flush()
                if items and items[-1][0] == "head" and not items[-1][1].endswith(tuple(".?!")) \
                        and items[-1][1].isupper():
                    items[-1] = ("head", f"{items[-1][1]} {text}")       # wrapped heading
                else:
                    items.append(("head", text))
                continue
            if kind == "bq":
                if cur and not cur.endswith(TERMINAL):
                    cur = f"{cur} {text}"
                else:
                    flush()
                    cur = text
                continue
            new_para = INDENT[0] <= x0 <= INDENT[1] or BULLET[0] <= x0 <= BULLET[1] or not cur
            if new_para:
                flush()
                cur = text
            elif cur.endswith("-") and text[:1].islower():
                cur = cur[:-1] + WRAP + text
            else:
                cur = f"{cur} {text}".strip()
    flush_epi()
    flush()
    # headings: ALL-CAPS from the page -> title case; TOC essay titles are already cased
    return [(k, title_case(t) if k == "head" and t.isupper() else t) for k, t in items]


def resolve_wraps(text):
    def pick(m):
        left, right = m.group(1), m.group(2)
        lw, rw = re.sub(r"\W", "", left).lower(), re.sub(r"\W", "", right).lower()
        if lw + rw in _dict or lw not in _dict or rw not in _dict:
            return left + right
        return f"{left}-{right}"
    return re.sub(r"([\w’']+)" + WRAP + r"([\w’']+)", pick, text)


def finish(text):
    text = resolve_wraps(text).replace(WRAP, "")
    for pat, rep in FIXUPS:
        text = re.sub(pat, rep, text)
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
    for s in sections:
        s.pop("essays")
    (HERE / "chapters.json").write_text(json.dumps(sections, indent=2), encoding="utf-8")
    print(f"wrote {OUT_MD.name} ({sum(len(m.split()) for m in md):,} words) and chapters.json")


if __name__ == "__main__":
    main()
