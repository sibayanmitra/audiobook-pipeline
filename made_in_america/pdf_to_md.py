#!/usr/bin/env python3
"""Convert Sam Walton: Made in America (Bantam, PDFCreator/Ghostscript PDF) to Markdown.

Usage:
    python pdf_to_md.py [INPUT.pdf]

Outputs (next to this script):
    ./made_in_america.md   `# ` heading per chapter, paragraphs as plain lines
    ./chapters.json        [{num, title, start_page, end_page}, ...]

Layout: one column, US Letter, Book Antiqua 12pt, no running headers or page numbers.
Paragraphs: first line at x~104, continuation lines at x=90; blank lines also separate them.
Each chapter page starts with its number (16-18pt) and the title in capitals (dropped: the
audiobook announces the chapter). The PDF has no outline, so titles come from CONTENTS.
Line-end hyphens are real compound-word hyphens (gung-ho, Wal-Mart), never word breaks.
Two small tables (p78, p93) are replaced by the same figures written as sentences.
Not narrated: title, copyright, contents, acknowledgments, the TIME ad on the last page.
"""
import glob
import json
import re
import sys
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
OUT_MD = HERE / "made_in_america.md"
TERMINAL = tuple(".!?:;\"')…—")
PARA_X = (100, 112)           # first line of a paragraph
KEEP_CAPS = {"CEO", "COO", "TV", "OTC", "NYSE", "ROTC", "IBM", "USA", "U.S.", "J.C.", "S.S.",
             "MBA", "PR", "NBC", "CBS", "ABC", "NFL", "UPC", "EDLP", "II", "III", "IV", "RV",
             "PX", "MIT", "OK", "K", "CPA", "P&G", "GE", "JCPenney", "TG&Y", "BB"}

# (first page, last page, title); pages are 1-based. Titles follow the CONTENTS page.
SECTIONS = [
    (6, 7, "Foreword"),
    (9, 15, "Chapter 1: Learning to Value a Dollar"),
    (16, 30, "Chapter 2: Starting on a Dime"),
    (31, 41, "Chapter 3: Bouncing Back"),
    (42, 55, "Chapter 4: Swimming Upstream"),
    (56, 63, "Chapter 5: Raising a Family"),
    (64, 73, "Chapter 6: Recruiting the Team"),
    (74, 85, "Chapter 7: Taking the Company Public"),
    (86, 97, "Chapter 8: Rolling Out the Formula"),
    (98, 109, "Chapter 9: Building the Partnership"),
    (110, 118, "Chapter 10: Stepping Back"),
    (119, 130, "Chapter 11: Creating a Culture"),
    (131, 141, "Chapter 12: Making the Customer Number One"),
    (142, 153, "Chapter 13: Meeting the Competition"),
    (154, 160, "Chapter 14: Expanding the Circles"),
    (161, 174, "Chapter 15: Thinking Small"),
    (175, 182, "Chapter 16: Giving Something Back"),
    (183, 186, "Chapter 17: Running a Successful Company: Ten Rules That Worked for Me"),
    (187, 192, "Chapter 18: Wanting to Leave a Legacy"),
    (193, 194, "A Postscript"),
    (195, 195, "Co-Author's Note"),
]

TABLES = {
    78: ("The 100 shares became 200 in May 1971, when the market price was 46 or 47 dollars, "
         "traded over the counter. 400 in March 1972, again at 46 or 47 dollars over the counter. "
         "800 in August 1975, at 23 dollars on the New York Stock Exchange. 1,600 in November 1980, "
         "at 50. 3,200 in June 1982, at 49 and seven-eighths. 6,400 in June 1983, at 81 and "
         "five-eighths. 12,800 in September 1985, at 49 and three-quarters. 25,600 in June 1987, "
         "at 66 and five-eighths. And 51,200 in June 1990, at 62 and a half."),
    93: ("In 1970, 32 stores and 31 million dollars in sales. In 1972, 51 stores and 78 million. "
         "In 1974, 78 stores and 168 million. In 1976, 125 stores and 340 million. In 1978, "
         "195 stores and 678 million. And in 1980, 276 stores and 1.2 billion dollars in sales."),
}

FIXUPS = [
    (r"^-{2,}\s*", "—"),                       # "----JACK SHEWMAKER" attribution dashes
    (r"(\w) —(\w)", r"\1—\2"),                 # "and —pure and simple" -> "and—pure"
    (r"(\w)— (\w)", r"\1—\2"),
    (r"(\w) —(?=\s)", r"\1—"),
    (r"(\w) —([,.;])", r"\1—\2"),
    (r", :", ","),                             # "HARRY CUNNINGHAM, : founded"
    (r"million'(\s|$)", r"million\1"),
    (r"\.\.\.", "… "),
    (r" \| ", " "),                            # stray bar from the PDF
    (r"^' (?=\")", ""),                       # stray quote before a quotation
    (r"\[([^\]]+)\]", r"(\1)"),                # editorial [glosses] read as an aside
    (r"\s+([,.;:!?])", r"\1"),
]


SIGNATURES = [                                 # applied after the capitals are softened
    ("Rob Walton Chairman, Wal-Mart Stores Bentonville, Arkansas May 1992",
     "Rob Walton, Chairman, Wal-Mart Stores, Bentonville, Arkansas, May 1992"),
    ("John Huey Atlanta, Georgia May 1992", "John Huey, Atlanta, Georgia, May 1992"),
]


def resolve_input(arg):
    if arg:
        return Path(arg)
    pdfs = sorted(glob.glob(str(HERE / "raw" / "*.pdf")))
    if not pdfs:
        sys.exit("No PDF found. Pass the path explicitly or place it in ./raw/")
    return Path(pdfs[0])


def is_caps(text):
    letters = re.sub(r"[^A-Za-z]", "", text)
    return len(letters) >= 2 and letters.isupper()


def title_word(w):
    core = re.sub(r"[^A-Za-z.&']", "", w)
    if core in KEEP_CAPS or len(re.sub(r"[^A-Za-z]", "", core)) < 2 or not core.isupper():
        return w
    if core in {"JR."}:
        return w.replace(core, "Jr.")
    return re.sub(r"[A-Za-z][A-Za-z']*", lambda m: m.group(0)[0] + m.group(0)[1:].lower(), w)


def soften_caps(text):
    """'—BUD WALTON' -> '—Bud Walton'; 'RULE 1: COMMIT to' -> 'Rule 1: Commit to'. ALL-CAPS
    runs read better as words; acronyms in KEEP_CAPS stay as they are."""
    return re.sub(r"[A-Z][A-Z.&']*[A-Z.]+'?S?\b|[A-Z]{2,}", lambda m: title_word(m.group(0)), text)


def page_lines(page):
    """-> [(x0, size, text)] for non-empty lines."""
    out = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            text = "".join(s["text"] for s in l["spans"]).strip()
            size = max((s["size"] for s in l["spans"] if s["text"].strip()), default=0)
            out.append((l["bbox"][0], size, text))
    return out


def section_paras(doc, first, last):
    paras, cur = [], ""

    def flush():
        nonlocal cur
        if cur.strip():
            paras.append(cur.strip())
        cur = ""

    for pno in range(first, last + 1):
        lines = page_lines(doc[pno - 1])
        if pno == first:                       # drop chapter number and the capitals title
            while lines and (not lines[0][2] or lines[0][1] >= 15 or is_caps(lines[0][2])):
                lines.pop(0)
        if pno in TABLES:
            rows = [i for i, (_, sz, t) in enumerate(lines) if 10.5 < sz < 11.5 and t]
            if rows:
                lines = lines[:rows[0]] + [(104, 12, ""), (104, 12, TABLES[pno]), (104, 12, "")] \
                    + lines[rows[-1] + 1:]
        for x0, size, text in lines:
            if not text:
                flush()
                continue
            if PARA_X[0] <= x0 <= PARA_X[1] or not cur:
                flush()
                cur = text
            elif cur.endswith("-"):
                cur += text                    # compound-word hyphen at a line end
            else:
                cur = f"{cur} {text}"
    flush()
    return paras


def finish(text):
    for pat, rep in FIXUPS:
        text = re.sub(pat, rep, text)
    if text.startswith("—"):
        text = "— " + soften_caps(text[1:].strip())
    else:
        text = soften_caps(text)
    for old, new in SIGNATURES:
        text = text.replace(old, new)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text if text.endswith(TERMINAL + (",",)) else text + "."


def main():
    src = resolve_input(sys.argv[1] if len(sys.argv) > 1 else None)
    doc = pymupdf.open(src)
    print(f"{src.name}: {len(doc)} pages, {len(SECTIONS)} sections")
    md, meta = [], []
    for i, (first, last, title) in enumerate(SECTIONS):
        md.append(f"# {title}\n")
        md += [finish(p) + "\n" for p in section_paras(doc, first, last)]
        meta.append({"num": i, "title": title, "start_page": first, "end_page": last})
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    (HERE / "chapters.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {OUT_MD.name} ({sum(len(m.split()) for m in md):,} words) and chapters.json")


if __name__ == "__main__":
    main()
