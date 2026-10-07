#!/usr/bin/env python3
"""Convert Confessions of an Advertising Man (two-column PDF) to Markdown + a chapter map.

Usage:
    python pdf_to_md.py [INPUT.pdf] [--keep-footnotes]

Outputs (next to this script):
    ./confessions.md  one `# ` heading per chapter, `## ` for in-chapter subheads
    ./chapters.json   [{num, label, title, start_page, end_page}, ...] (1-based, inclusive)

The PDF is laid out in two columns with a running header, page numbers and
bottom-of-column footnotes. Per page we drop the header/page number, put the
left column before the right, and repair paragraphs that wrap across columns or
pages. Footnotes are dropped unless --keep-footnotes is given (then they are
appended to the end of the chapter as "Footnote. ..." paragraphs).
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import pymupdf
from spellchecker import SpellChecker

HERE = Path(__file__).resolve().parent

# Geometry of the 595x842pt page (see the font/bbox dump in the README).
HEADER_MAX_Y = 40        # running header: "David Ogilvy - Confessions of an Advertising Man"
PAGE_NUM_MIN_Y = 800     # centered page number
FOOTNOTE_MIN_Y = 725     # footnotes sit at the bottom of a column
COLUMN_SPLIT_X = 298     # centre line between the two columns
COL_X0 = (43, 303)       # left edge of each column
INDENT_PT = 8            # first-line indent is ~17pt; anything over this starts a paragraph
BIG_FONT = 20            # chapter numeral / title
SUBHEAD_FONT = 13        # "I. Headlines", "Posters", "Food products" ...
SUPERSCRIPT_FONT = 7     # footnote markers in the running text

TERMINAL = tuple(".!?:;”\"')’…")  # a paragraph may end here
ROMAN = {"I": 1, "V": 5, "X": 10}

WRAP = "\x01"   # marks a "word-" + "continuation" line wrap until we can decide on the hyphen
_dict = SpellChecker()

# Book-specific repairs for OCR errors / layout artifacts in this PDF's text layer.
# (pattern, replacement) applied to each finished paragraph, in order.
FIXUPS = [
    (r"\btoo \u0442\u0438\u043f\u0443 levels", "too many levels"),
    (r"nun\u0441 dimittis", "nunc dimittis"),
    (r"\u0413\u2019m", "I\u2019m"),
    (r"\bpdtissiers\b", "p\u00e2tissiers"),
    (r"\bNew Yock\b", "New York"),
    (r"^\{1\)", "(1)"),
    (r"^1i\)", "(1)"),
    (r"\b(\w+?)1 (?=[a-z])", r"\1 "),               # stray footnote digit: "had1 been"
    (r"1,105\. free", "1,105 free"),
    (r"^As A CHILD\b", "As a child"),                       # small-caps chapter openers
    (r"^([A-Z]{3,})(?= [a-z])", lambda m: m.group(1).capitalize()),
    (r"waiting- for", "waiting for"),
    (r"\bRetaHing\b", "Retailing"),
    (r"\bun-toaded\b", "unloaded"),
    (r"\bmain-landers\b", "mainlanders"),
    (r"\bPep-?peridge\b", "Pepperidge"),
    (r"\bgobetweens\b", "go-betweens"),
    (r"\blatterday\b", "latter-day"),
    (r"\bgardemanger\b", "garde-manger"),
    (r"\s*Ipswich, Massachusetts\s+DAVID OGILVY\s*", " "),   # sign-off spliced into a sentence
    # chart: face-cream promise votes -> a spoken list
    (r"^FACE CREAM (.*?)\s*Smoothes Out Wrinkles From this voting",
     lambda m: "The promises tested for the face cream, in order of votes received: "
               + re.sub(r"\s*\*+\s*", ". ", m.group(1)).strip().rstrip(".") + ". Smoothes Out Wrinkles.\n\nFrom this voting"),
    # chart: Hill & Knowlton poll -> a spoken list
    (r"^YES (Religious leaders.*?\d+) Thus we see",
     lambda m: "Percentage answering yes: "
               + re.sub(r"\s+(\d+)%?\s*(?=[A-Z]|$)", r", \1; ", m.group(1)).strip().rstrip(";") + ".\n\nThus we see"),
]


def roman_to_int(s: str) -> int:
    total = 0
    for a, b in zip(s, s[1:] + " "):
        v = ROMAN[a]
        total += -v if b in ROMAN and ROMAN[b] > v else v
    return total


def resolve_input(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        return p if p.is_absolute() else Path.cwd() / p
    pdfs = sorted(glob.glob(str(HERE / "raw" / "*.pdf")))
    if not pdfs:
        sys.exit("No PDF found. Pass the path explicitly or place it in ./raw/")
    return Path(pdfs[0])


def build_chapters(doc: pymupdf.Document) -> list[dict]:
    """Top-level TOC entries -> chapters with page ranges."""
    entries = [(t.strip(), p) for lvl, t, p in doc.get_toc() if lvl == 1]
    chapters = []
    for i, (raw, page) in enumerate(entries):
        raw = re.sub(r"\s+", " ", raw)
        m = re.match(r"^([IVX]+)\s+(.*)$", raw)
        if m:
            num, title = roman_to_int(m.group(1)), m.group(2).strip()
            label = f"Chapter {num}"
        else:
            num, title, label = 0, raw, raw
        end = entries[i + 1][1] if i + 1 < len(entries) else len(doc)
        # a chapter ends on the page where the next begins (they share that page)
        chapters.append({"num": num, "label": label, "title": title,
                         "start_page": page, "end_page": end})
    return chapters


def span_text(span: dict) -> str:
    return span["text"]


def read_page_blocks(page: pymupdf.Page, keep_footnotes: bool):
    """Return (blocks, footnotes, heading_bottom) for one page.

    blocks: list of dicts {x0, y0, col, lines:[(x0, text)], heads:[str]} for body text
    heading_bottom: y of the chapter heading on this page, or None
    """
    blocks, footnotes = [], []
    heading_y0 = None
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        x0, y0, x1, y1 = b["bbox"]
        spans = [s for l in b["lines"] for s in l["spans"]]
        text = "".join(s["text"] for s in spans).strip()
        if not text:
            continue
        top = max(s["size"] for s in spans)
        if y0 < HEADER_MAX_Y and "Confessions of an Advertising Man" in text:
            continue
        if y0 >= PAGE_NUM_MIN_Y and re.fullmatch(r"\d+", text):
            continue
        if top >= BIG_FONT:
            heading_y0 = y0 if heading_y0 is None else min(heading_y0, y0)
            continue
        if re.fullmatch(r"[*∗\s]+", text):  # "* * *" section break on its own
            continue
        if y0 >= FOOTNOTE_MIN_Y and re.match(r"[*∗†‡]", text):
            if keep_footnotes:
                footnotes.append(re.sub(r"\s+", " ", re.sub(r"^[*∗\s]+", "", text)))
            continue

        lines, heads = [], []
        for l in b["lines"]:
            body, head = [], []
            for s in l["spans"]:
                if s["size"] < SUPERSCRIPT_FONT:          # footnote marker in running text
                    continue
                (head if s["size"] >= SUBHEAD_FONT else body).append(span_text(s))
            if head and "".join(head).strip():
                heads.append("".join(head).strip())
            line = "".join(body).strip()
            line = re.sub(r"^(?:[*∗]\s*){3}", "", line).strip()   # leading "* * *"
            if line:
                lines.append((l["bbox"][0], line))
        if not lines and not heads:
            continue
        cx = (x0 + x1) / 2
        blocks.append({"x0": x0, "y0": y0, "col": 0 if cx < COLUMN_SPLIT_X else 1,
                       "lines": lines, "heads": heads})
    return blocks, footnotes, heading_y0


def order_blocks(blocks: list[dict]) -> list[dict]:
    return sorted(blocks, key=lambda b: (b["col"], b["y0"]))


class Flow:
    """Accumulates paragraphs, repairing wraps across lines, columns and pages."""

    def __init__(self):
        self.items: list[tuple[str, str]] = []   # ("p"|"h", text)
        self.cur = ""

    def _flush(self):
        if self.cur.strip():
            self.items.append(("p", self.cur.strip()))
        self.cur = ""

    def head(self, text: str):
        self._flush()
        self.items.append(("h", text))

    def line(self, text: str, indented: bool):
        if not self.cur:
            self.cur = text
            return
        if indented and self.cur.endswith(TERMINAL):
            self._flush()
            self.cur = text
        elif self.cur.endswith("-") and text[:1].islower():
            self.cur = self.cur[:-1] + WRAP + text     # decided later by resolve_wraps()
        else:
            self.cur += " " + text

    def block_start(self):
        """A new block begins: it continues the previous paragraph only if that
        paragraph stopped mid-sentence (column or page wrap)."""
        if self.cur.endswith(TERMINAL):
            self._flush()

    def finish(self) -> list[tuple[str, str]]:
        self._flush()
        return self.items


def resolve_wraps(text: str) -> str:
    """"dis\x01inherited" -> "disinherited", but "self\x01advertisement" -> "self-advertisement"."""
    def pick(m):
        left, right = m.group(1), m.group(2)
        lw, rw = re.sub(r"\W", "", left).lower(), re.sub(r"\W", "", right).lower()
        if lw + rw in _dict or lw not in _dict or rw not in _dict:
            return left + right
        return f"{left}-{right}"
    return re.sub(r"([\w\u2019']+)" + WRAP + r"([\w\u2019']+)", pick, text)


def fix_paragraph(text: str) -> str:
    text = resolve_wraps(text).replace(WRAP, "")
    for pat, rep in FIXUPS:
        text = re.sub(pat, rep, text, flags=re.S)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def feed(flow: Flow, blocks: list[dict]):
    for b in blocks:
        for h in b["heads"]:
            flow.head(re.sub(r"^[IVX]+\.\s*", "", h).strip())
        if not b["lines"]:
            continue
        flow.block_start()
        for x0, text in b["lines"]:
            indented = x0 - COL_X0[b["col"]] > INDENT_PT
            flow.line(text, indented)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?")
    ap.add_argument("--keep-footnotes", action="store_true")
    args = ap.parse_args()

    src = resolve_input(args.pdf)
    doc = pymupdf.open(src)
    chapters = build_chapters(doc)
    print(f"{src.name}: {len(doc)} pages, {len(chapters)} chapters")

    flow_for = [Flow() for _ in chapters]
    notes_for = [[] for _ in chapters]
    start_page = {c["start_page"]: i for i, c in enumerate(chapters)}

    for pno in range(chapters[0]["start_page"], len(doc) + 1):
        blocks, notes, heading_y0 = read_page_blocks(doc[pno - 1], args.keep_footnotes)
        # chapter this page's text belongs to (before any heading on the page)
        idx = max(i for i, c in enumerate(chapters) if c["start_page"] <= pno)
        if pno in start_page and heading_y0 is not None:
            new = start_page[pno]
            above = [b for b in blocks if b["y0"] < heading_y0]
            below = [b for b in blocks if b["y0"] >= heading_y0]
            if new > 0:
                feed(flow_for[new - 1], order_blocks(above))
            else:
                below = above + below
            feed(flow_for[new], order_blocks(below))
            notes_for[idx].extend(notes)
        else:
            feed(flow_for[idx], order_blocks(blocks))
            notes_for[idx].extend(notes)

    md = []
    for c, flow, notes in zip(chapters, flow_for, notes_for):
        head = c["title"] if c["num"] == 0 else f"{c['label']}: {c['title']}"
        md.append(f"# {head}\n")
        for kind, text in flow.finish():
            md.append(("## " if kind == "h" else "") + (text if kind == "h" else fix_paragraph(text)) + "\n")
        for n in notes:
            md.append(f"Footnote. {n}\n")

    out_md, out_json = HERE / "confessions.md", HERE / "chapters.json"
    out_md.write_text("\n".join(md), encoding="utf-8")
    out_json.write_text(json.dumps(chapters, indent=2), encoding="utf-8")
    words = sum(len(m.split()) for m in md)
    print(f"wrote {out_md.name} ({words:,} words) and {out_json.name}")
    for c in chapters:
        print(f"  {c['label']:<11} p{c['start_page']:>2}-{c['end_page']:<2} {c['title']}")


if __name__ == "__main__":
    main()
