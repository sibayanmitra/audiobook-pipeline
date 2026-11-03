#!/usr/bin/env python3
"""Convert The Goal (Goldratt & Cox, 30th anniversary edition, calibre PDF) to Markdown.

Usage:
    python pdf_to_md.py [INPUT.pdf]

Outputs (next to this script):
    ./the_goal.md       `# ` heading per section, `## ` for the essay's subheadings
    ./chapters.json     [{num, label, title, start_page, end_page}, ...]

Layout: one column, no running headers or page numbers. A paragraph's first line is
indented (x~78) and its continuation lines are not (x~57), which also tells us whether a
page starts a new paragraph or continues the last one. Sections come from the PDF's TOC;
cover, copyright, contents and "About the Author" are not narrated. Chapter numbers
(26.6pt) are dropped because the audiobook announces each chapter itself, superscript
note markers (10.8pt) are dropped, and the endnotes of "Standing on the Shoulders of
Giants" are cut off.
"""
import glob
import json
import re
import sys
from pathlib import Path

import pymupdf
from spellchecker import SpellChecker

HERE = Path(__file__).resolve().parent
OUT_MD = HERE / "the_goal.md"

NARRATE = re.compile(r"^(Introduction|Introduction to the First Edition|Chapter \d+|"
                     r"Standing on the Shoulders of Giants|Five Focusing Steps)$")
BODY_X0 = 66            # continuation lines start at ~57, indented first lines at ~78
TITLE_FONT = 19.5       # chapter numbers / section titles (20.2pt and up)
SUBHEAD_FONT = 15.5     # essay subheadings, figure labels (16.6pt)
SUPERSCRIPT_FONT = 12   # note markers (10.8pt); body text is 14.4pt
TERMINAL = tuple(".!?:;”\"')’…—")
WRAP = "\x01"
_dict = SpellChecker()

# (pattern, replacement) repairs applied to each finished paragraph.
FIXUPS = [
    (r"\s*…\s*", "… "),                       # "Hot … Very Hot" -> even spacing
    # words run together or misspelt in this PDF's text layer (dialect like "somthin'" is kept)
    (r"\bplus2 hours", "plus 2 hours"), (r"\bfiight\b", "flight"), (r"\bstiffiy\b", "stiffly"),
    (r"\bActi vating\b", "Activating"), (r"\bincloning\b", "in cloning"),
    (r"\bhand-inhand\b", "hand-in-hand"), (r"\bwork-inprocess\b", "work-in-process"),
    (r"\bawhiteboard\b", "a whiteboard"), (r"\bhowbad\b", "how bad"),
    (r"\bincreasesales\b", "increase sales"), (r"\bintendto\b", "intend to"),
    (r"\bnewsuper computer\b", "new supercomputer"), (r"\boffduty\b", "off-duty"),
    (r"\bofoffice\b", "office"), (r"\bpackstraps\b", "pack straps"),
    (r"\bcost-perpart\b", "cost-per-part"), (r"\bplateglass\b", "plate glass"),
    (r"\bpreventsoverproduction\b", "prevents overproduction"),
    (r"\bseatof-the-pants\b", "seat-of-the-pants"), (r"\bthecomputer\b", "the computer"),
    (r"\bbreifcase\b", "briefcase"), (r"\bcatastrophies\b", "catastrophes"),
    (r"\birrelevent\b", "irrelevant"), (r"\bmillenium\b", "millennium"),
    (r"\bhandside\b", "hand side"), (r"\bshe crys\b", "she cries"),
]


def resolve_input(arg):
    if arg:
        return Path(arg)
    pdfs = sorted(glob.glob(str(HERE / "raw" / "*.pdf")))
    if not pdfs:
        sys.exit("No PDF found. Pass the path explicitly or place it in ./raw/")
    return Path(pdfs[0])


def build_sections(doc):
    toc = [(re.sub(r"\s+", " ", t).strip(), p) for lvl, t, p in doc.get_toc() if lvl == 1]
    sections = []
    for i, (title, page) in enumerate(toc):
        if not NARRATE.match(title):
            continue
        end = toc[i + 1][1] - 1 if i + 1 < len(toc) else len(doc)
        m = re.match(r"Chapter (\d+)", title)
        sections.append({"num": int(m.group(1)) if m else 0, "label": title, "title": title,
                         "start_page": page, "end_page": end})
    return sections


def resolve_wraps(text):
    def pick(m):
        left, right = m.group(1), m.group(2)
        lw, rw = re.sub(r"\W", "", left).lower(), re.sub(r"\W", "", right).lower()
        if lw + rw in _dict or lw not in _dict or rw not in _dict:
            return left + right
        return f"{left}-{right}"
    return re.sub(r"([\w’']+)" + WRAP + r"([\w’']+)", pick, text)


def page_lines(page):
    """-> [(x0, kind, text)], kind = 'title' | 'head' | 'body'."""
    out = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            spans = [s for s in l["spans"] if s["size"] >= SUPERSCRIPT_FONT]
            text = "".join(s["text"] for s in spans).strip()
            if not text:
                continue
            size = max(s["size"] for s in spans)
            kind = "title" if size >= TITLE_FONT else "head" if size >= SUBHEAD_FONT else "body"
            out.append((l["bbox"][0], kind, text))
    return out


def section_items(doc, sec):
    items, cur, prev_x = [], "", None

    def flush():
        nonlocal cur
        if cur:
            items.append(("para", cur))
        cur = ""

    for pno in range(sec["start_page"], sec["end_page"] + 1):
        for x0, kind, text in page_lines(doc[pno - 1]):
            if kind == "title":
                continue
            if re.fullmatch(r"[\d\s]+", text) or re.fullmatch(r"Figure \d+", text):
                continue                                    # table digits, figure labels
            if sec["label"].startswith("Standing") and re.match(r"1 \S", text) \
                    and pno >= sec["end_page"] - 2:
                flush()
                return items                                # endnotes ("1 http://...") begin: stop
            if kind == "head":
                if sec["label"] == "Five Focusing Steps":   # the steps themselves, 16.6pt
                    if re.match(r"\d\. ", text):
                        flush()
                        cur = text
                    else:
                        cur = f"{cur} {text}".strip()
                    continue
                flush()
                items.append(("head", text))
                prev_x = None
                continue
            if re.fullmatch(r"STEP \d\.?", text):
                flush()
                items.append(("para", text))
                prev_x = None
                continue
            new_para = x0 > BODY_X0 and (not cur or cur.endswith(TERMINAL) or abs(x0 - (prev_x or 0)) > 5)
            if new_para:
                flush()
                cur = text
            elif cur.endswith("-") and text[:1].islower():
                cur = cur[:-1] + WRAP + text
            else:
                cur = f"{cur} {text}".strip()
            prev_x = x0
    flush()
    return items


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
    (HERE / "chapters.json").write_text(json.dumps(sections, indent=2), encoding="utf-8")
    print(f"wrote {OUT_MD.name} ({sum(len(m.split()) for m in md):,} words) and chapters.json")


if __name__ == "__main__":
    main()
