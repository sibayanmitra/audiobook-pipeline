#!/usr/bin/env python3
"""Convert a PDF to Markdown using pymupdf4llm.

Usage:
    python pdf_to_md.py [INPUT.pdf] [OUTPUT.md]

Defaults:
    INPUT  = ./raw/One_Up_On_WallStreet.pdf  (first *.pdf in ./raw if not found)
    OUTPUT = ./One_Up_On_WallStreet.md
"""
import sys
import glob
import time
from pathlib import Path

import pymupdf4llm

HERE = Path(__file__).resolve().parent


def resolve_input(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        if not p.is_absolute():
            p = HERE / p
        return p
    default = HERE / "raw" / "One_Up_On_WallStreet.pdf"
    if default.exists():
        return default
    # fall back to the first PDF in raw/
    pdfs = sorted(glob.glob(str(HERE / "raw" / "*.pdf")))
    if not pdfs:
        sys.exit("No PDF found. Pass the path explicitly or place it in ./raw/")
    return Path(pdfs[0])


def main() -> None:
    in_path = resolve_input(sys.argv[1] if len(sys.argv) > 1 else None)
    out_path = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "One_Up_On_WallStreet.md"
    )
    if not out_path.is_absolute():
        out_path = HERE / out_path

    if not in_path.exists():
        sys.exit(f"Input not found: {in_path}")

    print(f"Converting: {in_path}")
    print(f"     -> md: {out_path}")
    t0 = time.time()
    # page_chunks=False returns one big markdown string for the whole doc.
    md = pymupdf4llm.to_markdown(str(in_path), show_progress=True)
    out_path.write_text(md, encoding="utf-8")
    dt = time.time() - t0
    chars = len(md)
    print(f"Done in {dt:.1f}s | {chars:,} chars | ~{chars // 4:,} tokens (rough)")


if __name__ == "__main__":
    main()
