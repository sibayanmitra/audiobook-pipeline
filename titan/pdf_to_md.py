#!/usr/bin/env python3
"""Convert Titan.pdf to Markdown + a chapter map built from the PDF's own TOC.

Usage:
    python pdf_to_md.py [INPUT.pdf]

Outputs:
    ./Titan.md       one markdown blob with <!--PAGE:n--> markers
    ./chapters.json  [{num, title, start_page, end_page}, ...] (1-based, inclusive)

The book's back matter (NOTES, BIBLIOGRAPHY, ...) is excluded — endnotes are
citation soup and unreadable as audio.
"""
import glob
import json
import re
import sys
import time
from pathlib import Path

import pymupdf
import pymupdf4llm

HERE = Path(__file__).resolve().parent

# TOC entries after this one are back matter and are not narrated.
STOP_AT = "NOTES"
# Front matter we do narrate; everything before FOREWORD is title/dedication/blurbs.
START_AT = "FOREWORD"


def resolve_input(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        return p if p.is_absolute() else HERE / p
    default = HERE / "raw" / "Titan.pdf"
    if default.exists():
        return default
    pdfs = sorted(glob.glob(str(HERE / "raw" / "*.pdf")))
    if not pdfs:
        sys.exit("No PDF found. Pass the path explicitly or place it in ./raw/")
    return Path(pdfs[0])


def build_chapters(doc: pymupdf.Document) -> list:
    """Slice the TOC into narratable sections with page ranges."""
    toc = [(lvl, title.strip(), page) for lvl, title, page in doc.get_toc()]
    names = [t[1].upper() for t in toc]

    try:
        first = names.index(START_AT)
    except ValueError:
        first = 0
    try:
        last = names.index(STOP_AT)
    except ValueError:
        last = len(toc)

    entries = toc[first:last]
    chapters = []
    for i, (_, title, page) in enumerate(entries):
        end = entries[i + 1][2] - 1 if i + 1 < len(entries) else toc[last][2] - 1
        m = re.match(r"CHAPTER\s+(\d+)", title, re.I)
        num = int(m.group(1)) if m else 0
        chapters.append(
            {
                "num": num,
                "label": title,
                "title": chapter_title(doc, page, title),
                "start_page": page,
                "end_page": end,
            }
        )

    # front matter keeps document order ahead of chapter 1
    for i, ch in enumerate(chapters):
        if ch["num"] == 0:
            ch["num"] = -len(chapters) + i
    return chapters


def chapter_title(doc: pymupdf.Document, page: int, label: str) -> str:
    """The line under 'CHAPTER N' on the opening page is the chapter title."""
    lines = [l.strip() for l in doc[page - 1].get_text().splitlines() if l.strip()]
    if not lines:
        return label.title()
    head = lines[0].upper()
    if head.startswith("CHAPTER") or head.startswith("PRELUDE"):
        if len(lines) > 1:
            return lines[1].strip(" -–—:")
    return label.title()


def main() -> None:
    in_path = resolve_input(sys.argv[1] if len(sys.argv) > 1 else None)
    if not in_path.exists():
        sys.exit(f"Input not found: {in_path}")

    doc = pymupdf.open(str(in_path))
    chapters = build_chapters(doc)
    lo = min(c["start_page"] for c in chapters)
    hi = max(c["end_page"] for c in chapters)

    print(f"Converting: {in_path}")
    print(f"  narrating pages {lo}-{hi} of {doc.page_count} | {len(chapters)} sections")
    t0 = time.time()

    pages = pymupdf4llm.to_markdown(
        str(in_path),
        pages=list(range(lo - 1, hi)),
        page_chunks=True,
        show_progress=False,
    )

    parts = []
    for chunk in pages:
        pno = chunk["metadata"]["page_number"]
        parts.append(f"\n\n<!--PAGE:{pno}-->\n\n{chunk['text']}")
    md = "".join(parts)

    (HERE / "Titan.md").write_text(md, encoding="utf-8")
    (HERE / "chapters.json").write_text(json.dumps(chapters, indent=2), encoding="utf-8")

    dt = time.time() - t0
    print(f"Done in {dt:.1f}s | {len(md):,} chars")
    for c in chapters:
        print(f"  {c['num']:>3}  p{c['start_page']:>3}-{c['end_page']:<3}  {c['title']}")


if __name__ == "__main__":
    main()
