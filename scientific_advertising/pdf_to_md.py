#!/usr/bin/env python3
"""Convert Scientific Advertising (Claude C. Hopkins, 1923; Galletti edition) to Markdown.

Usage:
    python pdf_to_md.py [INPUT.pdf]

Outputs (next to this script):
    ./scientific_advertising.md   `# ` heading per chapter ("# Chapter 2: Just Salesmanship")
    ./chapters.json               [{num, label, title, start_page, end_page}, ...]

This edition has no PDF bookmarks, so chapters are found from the 28pt "Chapter N" headings.
Every page carries a "Carl Galletti - N - ScientificAdvertising.com" footer (dropped), and the
real text stops at the "--- The End ---" line; the pages after it are the editor's product
catalogue and are never narrated.
"""
import glob
import json
import re
import sys
from pathlib import Path

import pymupdf
from spellchecker import SpellChecker

HERE = Path(__file__).resolve().parent
OUT_MD = HERE / "scientific_advertising.md"

FOOTER_MIN_Y = 715
HEADING_FONT = 26
TERMINAL = tuple(".!?:;”\"')’…")
WRAP = "\x01"
END_MARK = re.compile(r"-+\s*The End\s*-+", re.I)
_dict = SpellChecker()

FIXUPS = [   # (pattern, replacement) repairs per paragraph
    (r"\bhe is a tought efficiency\b", "he is taught efficiency"),
]


def resolve_input(arg):
    if arg:
        return Path(arg)
    pdfs = sorted(glob.glob(str(HERE / "raw" / "*.pdf")))
    if not pdfs:
        sys.exit("No PDF found. Pass the path explicitly or place it in ./raw/")
    return Path(pdfs[0])


def resolve_wraps(text):
    def pick(m):
        left, right = m.group(1), m.group(2)
        lw, rw = re.sub(r"\W", "", left).lower(), re.sub(r"\W", "", right).lower()
        if lw + rw in _dict or lw not in _dict or rw not in _dict:
            return left + right
        return f"{left}-{right}"
    return re.sub(r"([\w’']+)" + WRAP + r"([\w’']+)", pick, text)


def block_text(block):
    out = ""
    for l in block["lines"]:
        t = "".join(s["text"] for s in l["spans"]).strip()
        if not t:
            continue
        out = out[:-1] + WRAP + t if out.endswith("-") and t[:1].islower() else f"{out} {t}".strip()
    return out


def main():
    src = resolve_input(sys.argv[1] if len(sys.argv) > 1 else None)
    doc = pymupdf.open(src)
    chapters, paras, cur_ch, done = [], [], None, False

    def flush_para(cur):
        if cur:
            for pat, rep in FIXUPS:
                cur = re.sub(pat, rep, cur)
            para = re.sub(r"[ \t]{2,}", " ", resolve_wraps(cur).replace(WRAP, "")).strip()
            paras[-1].append(para if para.endswith(TERMINAL) else para + ".")

    cur = ""
    for pno in range(2, len(doc) + 1):          # page 1 is the cover
        if done:
            break
        blocks = [b for b in doc[pno - 1].get_text("dict")["blocks"] if b["type"] == 0]
        blocks.sort(key=lambda b: b["bbox"][1])
        head_lines = []
        for b in blocks:
            text = block_text(b)
            if not text or b["bbox"][1] >= FOOTER_MIN_Y:
                continue
            size = max(s["size"] for l in b["lines"] for s in l["spans"])
            if size >= HEADING_FONT:
                head_lines.append(text)
                continue
            if head_lines:                       # heading blocks end where body text begins
                flush_para(cur); cur = ""
                m = re.match(r"Chapter (\d+)\s*(.*)", " ".join(head_lines))
                cur_ch = {"num": int(m.group(1)), "label": f"Chapter {m.group(1)}",
                          "title": m.group(2).strip(), "start_page": pno, "end_page": pno}
                chapters.append(cur_ch); paras.append([])
                head_lines = []
            if cur_ch is None:
                continue
            end = END_MARK.search(text)
            if end:
                text, done = text[:end.start()].strip(), True
            # blocks are paragraphs; one that follows an unfinished sentence continues it
            # (a paragraph running over a page break)
            if cur and not cur.endswith(TERMINAL):
                cur = f"{cur} {text}"
            else:
                flush_para(cur); cur = text
            cur_ch["end_page"] = pno
            if done:
                break
    flush_para(cur)

    md = []
    for ch, ps in zip(chapters, paras):
        md.append(f"# {ch['label']}: {ch['title']}\n")
        md.extend(p + "\n" for p in ps if p)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    (HERE / "chapters.json").write_text(json.dumps(chapters, indent=2), encoding="utf-8")
    print(f"wrote {OUT_MD.name} ({sum(len(m.split()) for m in md):,} words) and chapters.json")
    for c in chapters:
        print(f"  {c['label']:<11} p{c['start_page']:>3}-{c['end_page']:<3} {c['title']}")


if __name__ == "__main__":
    main()
