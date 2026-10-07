#!/usr/bin/env python3
"""Convert My Life in Advertising (Claude C. Hopkins, 1927) to Markdown + a chapter map.

Usage:
    python pdf_to_md.py [INPUT.pdf]

Outputs (next to this script):
    ./my_life_in_advertising.md   `# ` heading per chapter ("# Chapter 1: Early Influences")
    ./chapters.json               [{num, label, title, start_page, end_page}, ...]

Layout: one column, one text block per page, every line at the same left margin, and a
whitespace-only line between paragraphs. Chapters come from the PDF's TOC; each chapter starts
on a fresh page whose first line(s) are the printed heading, which we drop (the audiobook
announces the chapter itself). A paragraph that runs over a page break is rejoined.
"""
import glob
import json
import re
import sys
from pathlib import Path

import pymupdf
from spellchecker import SpellChecker

HERE = Path(__file__).resolve().parent
OUT_MD = HERE / "my_life_in_advertising.md"

SKIP_TOC = {"my life in advertising", "contents"}
TERMINAL = tuple(".!?:;”\"')’…")
WRAP = "\x01"
_dict = SpellChecker()

# (pattern, replacement) repairs applied to each finished paragraph.
FIXUPS = [
    (r"\s*\bThe End\b\s*$", ""),
    (r"(?<=production\.) 113 106$", ""),            # stray figures left over from a table
    (r"\uff02", "\u201d"),                           # fullwidth quotation mark
    (r"\binsiduous\b", "insidious"),
]


def resolve_input(arg):
    if arg:
        return Path(arg)
    pdfs = sorted(glob.glob(str(HERE / "raw" / "*.pdf")))
    if not pdfs:
        sys.exit("No PDF found. Pass the path explicitly or place it in ./raw/")
    return Path(pdfs[0])


def build_chapters(doc):
    entries = [(re.sub(r"\s+", " ", t).strip(), p) for lvl, t, p in doc.get_toc() if lvl == 1]
    entries = [(t, p) for t, p in entries if t.lower() not in SKIP_TOC]
    chapters = []
    for i, (raw, page) in enumerate(entries):
        m = re.match(r"^Ch (\d+) (.*)$", raw)
        num, title = (int(m.group(1)), m.group(2).strip()) if m else (0, raw)
        end = entries[i + 1][1] - 1 if i + 1 < len(entries) else len(doc)
        chapters.append({"num": num, "label": f"Chapter {num}" if m else raw, "title": title,
                         "start_page": page, "end_page": end})
    return chapters


def page_lines(page):
    """-> [(text, width)] for every line on the page (blank lines have empty text)."""
    out = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            text = "".join(s["text"] for s in l["spans"]).strip()
            out.append((text, l["bbox"][2] - l["bbox"][0]))
    return out


def resolve_wraps(text):
    def pick(m):
        left, right = m.group(1), m.group(2)
        lw, rw = re.sub(r"\W", "", left).lower(), re.sub(r"\W", "", right).lower()
        if lw + rw in _dict or lw not in _dict or rw not in _dict:
            return left + right
        return f"{left}-{right}"
    return re.sub(r"([\w’']+)" + WRAP + r"([\w’']+)", pick, text)


def chapter_paragraphs(doc, ch):
    paras, cur = [], ""
    for pno in range(ch["start_page"], ch["end_page"] + 1):
        lines = page_lines(doc[pno - 1])
        if pno == ch["start_page"]:                      # drop the printed heading (up to first blank)
            while lines and lines[0][0]:
                lines.pop(0)
        widest = max((w for t, w in lines if t), default=0)
        last_width = 0
        for text, width in lines:
            if not text:
                if cur:
                    paras.append(cur)
                cur = ""
                continue
            if cur.endswith("-") and text[:1].islower():
                cur = cur[:-1] + WRAP + text
            else:
                cur = f"{cur} {text}".strip()
            last_width = width
        # a short last line that ends a sentence = paragraph ended at the page break
        if cur and cur.endswith(TERMINAL) and last_width < 0.8 * widest:
            paras.append(cur)
            cur = ""
    if cur:
        paras.append(cur)
    out = []
    for p in paras:
        p = resolve_wraps(p).replace(WRAP, "")
        for pat, rep in FIXUPS:
            p = re.sub(pat, rep, p)
        p = re.sub(r"[ \t]{2,}", " ", p).strip()
        if p and not p.endswith(TERMINAL):               # the source drops a few full stops
            p += "."
        if p:
            out.append(p)
    return out


def main():
    src = resolve_input(sys.argv[1] if len(sys.argv) > 1 else None)
    doc = pymupdf.open(src)
    chapters = build_chapters(doc)
    print(f"{src.name}: {len(doc)} pages, {len(chapters)} chapters")
    md = []
    for ch in chapters:
        head = ch["title"] if ch["num"] == 0 else f"{ch['label']}: {ch['title']}"
        md.append(f"# {head}\n")
        md.extend(p + "\n" for p in chapter_paragraphs(doc, ch))
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    (HERE / "chapters.json").write_text(json.dumps(chapters, indent=2), encoding="utf-8")
    print(f"wrote {OUT_MD.name} ({sum(len(m.split()) for m in md):,} words) and chapters.json")
    for c in chapters:
        print(f"  {c['label']:<11} p{c['start_page']:>3}-{c['end_page']:<3} {c['title']}")


if __name__ == "__main__":
    main()
