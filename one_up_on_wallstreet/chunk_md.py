#!/usr/bin/env python3
"""Chunk a Markdown file into semantically-coherent pieces for RAG/embeddings.

Strategy (keeps meaning intact):
  1. Split on Markdown headers (#, ##, ###) so chapter/section boundaries are
     respected and each chunk keeps its heading path as metadata.
  2. Within each section, recursively split on paragraph -> sentence -> word
     boundaries, sized by *tokens* (tiktoken), with overlap so context bleeds
     across adjacent chunks.

Usage:
    python chunk_md.py [INPUT.md] [--out OUTPUT.jsonl]
                       [--chunk-tokens 800] [--overlap 120]

Output: JSONL, one chunk per line:
    {"id", "text", "n_tokens", "headers": {...}, "source"}
"""
import argparse
import json
from pathlib import Path

import tiktoken
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

HERE = Path(__file__).resolve().parent
ENC = tiktoken.get_encoding("cl100k_base")


def n_tokens(text: str) -> int:
    return len(ENC.encode(text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default=str(HERE / "One_Up_On_WallStreet.md"))
    ap.add_argument("--out", default=str(HERE / "One_Up_On_WallStreet.chunks.jsonl"))
    ap.add_argument("--chunk-tokens", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=120)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = HERE / in_path
    if not in_path.exists():
        raise SystemExit(f"Markdown not found: {in_path}  (run pdf_to_md.py first)")

    md = in_path.read_text(encoding="utf-8")
    print(f"Loaded {in_path.name}: {len(md):,} chars, ~{n_tokens(md):,} tokens")

    # 1) split on headers, carrying the heading hierarchy into metadata
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    sections = header_splitter.split_text(md)
    print(f"Header sections: {len(sections)}")

    # 2) recursively split each section by tokens, with overlap
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=args.chunk_tokens,
        chunk_overlap=args.overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
    )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = HERE / out_path

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for sec in sections:
            for piece in splitter.split_text(sec.page_content):
                piece = piece.strip()
                if not piece:
                    continue
                rec = {
                    "id": f"one_up_on_wallstreet-{n:05d}",
                    "text": piece,
                    "n_tokens": n_tokens(piece),
                    "headers": sec.metadata,
                    "source": in_path.name,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1

    toks = [json.loads(l)["n_tokens"] for l in out_path.read_text("utf-8").splitlines()]
    avg = sum(toks) / len(toks) if toks else 0
    print(f"Wrote {n:,} chunks -> {out_path}")
    print(f"Tokens/chunk: avg {avg:.0f}, min {min(toks)}, max {max(toks)}")


if __name__ == "__main__":
    main()
