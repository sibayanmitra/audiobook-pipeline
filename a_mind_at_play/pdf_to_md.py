#!/usr/bin/env python3
"""Convert a PDF to Markdown using pymupdf4llm.

Usage:
    python pdf_to_md.py [INPUT.pdf] [OUTPUT.md]

Defaults:
    INPUT  = ./raw/  (first *.pdf found there)
    OUTPUT = ./a_mind_at_play.md
"""
import sys
import glob
import time
from pathlib import Path

import pymupdf4llm

HERE = Path(__file__).resolve().parent


def resolve_input(arg):
    if arg:
        p = Path(arg)
        if not p.is_absolute():
            p = HERE / p
        return p
    pdfs = sorted(glob.glob(str(HERE / "raw" / "*.pdf")))
    if not pdfs:
        sys.exit("No PDF found in ./raw/  — place the book PDF there or pass the path explicitly.")
    return Path(pdfs[0])


def main():
    in_path  = resolve_input(sys.argv[1] if len(sys.argv) > 1 else None)
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "a_mind_at_play.md"
    if not out_path.is_absolute():
        out_path = HERE / out_path

    if not in_path.exists():
        sys.exit(f"Input not found: {in_path}")

    print(f"Converting: {in_path}")
    print(f"       → md: {out_path}")
    t0 = time.time()
    md = pymupdf4llm.to_markdown(str(in_path), show_progress=True)
    out_path.write_text(md, encoding="utf-8")
    dt = time.time() - t0
    print(f"Done in {dt:.1f}s | {len(md):,} chars | ~{len(md)//4:,} tokens")


if __name__ == "__main__":
    main()
