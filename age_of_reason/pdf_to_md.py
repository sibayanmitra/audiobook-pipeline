#!/usr/bin/env python3
"""Convert The Age of Reason (Sartre, tr. Eric Sutton, Hamish Hamilton 1947; a scanned PDF
with no text layer) to Markdown via Tesseract OCR.

Usage:
    python pdf_to_md.py [INPUT.pdf]

Steps:
    1. each page's embedded scan (2840x4252, 1-bit, ~600 dpi) is OCR'd once with
       `tesseract --psm 4` into ./ocr/pNNN.tsv (cached; delete ./ocr to redo)
    2. lines are rebuilt from the word boxes; a paragraph starts where a line is indented
       against that page's own left margin (scans drift a few pixels page to page)
    3. "CHAPTER N" lines (often misread: CIIAPTER, GHAPTER, I!) split the book into 18
       chapters; page numbers, the title page, imprint and library slips are dropped
    4. quote marks are removed (they are not spoken, and OCR garbles them most), words
       hyphenated across lines are re-joined, and recurring OCR misreads are repaired

Outputs (next to this script): ./age_of_reason.md, ./chapters.json
"""
import csv
import glob
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf
from spellchecker import SpellChecker

HERE = Path(__file__).resolve().parent
OUT_MD = HERE / "age_of_reason.md"
OCR_DIR = HERE / "ocr"
FIRST_PAGE, LAST_PAGE = 3, 356          # 1-2 title/imprint, 357-359 blank + library slips
INDENT_PX = 50                           # first-line indent is ~90 px at 600 dpi
CHAPTER = re.compile(r"^[CG][HI]{1,2}APTER\s+([IVXL1!]+)$")
TERMINAL = tuple(".!?:;…—")
WRAP = "\x01"
_dict = SpellChecker()

# word-level OCR repairs (applied to each finished paragraph)
FIXUPS = [
    # quote marks misread as other symbols
    (r"(^|\s)[*|~^{}\[\]]+(?=\S)", r"\1"),                 # *No, / **How -> No, / How
    (r"\s+\|+\.?(?=\s|$)", ""), (r"(?<=\S)[*|~^{}\[\]]+", ""), (r"(^|\s)[*|~^{}\[\]]+(\s|$)", r"\1"),
    (r"^c[ce] (?=[A-Z])", ""),                                # “ read as "ce"
    (r"([,.!])\?(?=\s|$)", r"\1"),                           # ” read as ? : "Yes,?" -> "Yes,"
    # capital I misread as T / 7 / 1 / Y / P
    (r"\b[T7Y](?=[’'](?:m|ll|ve|d)\b)", "I"), (r"\b[T7]t\b", "It"), (r"\b[T7]t’s\b", "It’s"),
    (r"\bTf\b", "If"), (r"\b(?:Pll|Pil|Tll)\b", "I’ll"), (r"(?<=[.!?…] )ll\b", "I’ll"), (r"\bTn\b", "In"), (r"\bTs\b", "Is"), (r"\bPll\b", "I’ll"), (r"\bVes\b", "Yes"),
    (r"(?<![\d.,£$])\b1\b(?![.,]?\d)", "I"), (r"\bI1\b", "I"), (r"\b1(?=[a-z]{2,}\b)", ""),
    (r"\b(\d)o(?=th\b)", r"\g<1>0"),                          # 2oth -> 20th
    (r"\b([CG])G(?=[a-z])", "C"),
    (r"\bI!(?=[’']m\b)", "I"), (r"!(?=[’']m\b)", "I"),       # !’m -> I’m
    (r"(?<![A-Za-z])(?:IT|TI|Il|1)(?= [a-z])", "I"),          # "IT know" / "TI did" -> I
    (r"\bCGan\b", "Can"), (r"\bG\.N\.T\.", "C.N.T."), (r"\bGhemins\b", "Chemins"),
    (r"\bHALF-wAy\b", "Half-way"),
    # one-off misreads found by reviewing the unknown-word list
    (r"\baatanly\b", "saintly"),
    (r"\bacard\b", "a card"),
    (r"\badnft\b", "adrift"),
    (r"\baetested\b", "detested"),
    (r"\bahumiliating\b", "a humiliating"),
    (r"\baliost\b", "almost"),
    (r"\balout\b", "about"),
    (r"\baman\b", "a man"),
    (r"\bassing\b", "passing"),
    (r"\bavich\b", "Ivich"),
    (r"\bbttle\b", "little"),
    (r"\bchese\b", "these"),
    (r"\bconsciousriess\b", "consciousness"),
    (r"\bcordiatly\b", "cordially"),
    (r"\bdeepty\b", "deeply"),
    (r"\bdemneanour\b", "demeanour"),
    (r"\bdisciplesif\b", "disciples if"),
    (r"\bexzsted\b", "existed"),
    (r"\bfagade\b", "façade"),
    (r"\bfagades\b", "façades"),
    (r"\bfina\b", "find"),
    (r"\bfoolisk\b", "foolish"),
    (r"\bhife\b", "life"),
    (r"\bhitn\b", "him"),
    (r"\bifyou\b", "if you"),
    (r"\billhumour\b", "ill-humour"),
    (r"\bingeriuous\b", "ingenuous"),
    (r"\bithe\b", "the"),
    (r"\bkeer\b", "keep"),
    (r"\bker\b", "her"),
    (r"\blifz\b", "life"),
    (r"\blitue\b", "little"),
    (r"\blke\b", "like"),
    (r"\bmakce\b", "make"),
    (r"\bmunths\b", "months"),
    (r"\bnie\b", "me"),
    (r"\bnonchalauatly\b", "nonchalantly"),
    (r"\bnzrrow\b", "narrow"),
    (r"\bofa\b", "of a"),
    (r"\book\b", "look"),
    (r"\bpatchof\b", "patch of"),
    (r"\bprece\b", "piece"),
    (r"\bquestior\b", "question"),
    (r"\braore\b", "more"),
    (r"\breadirg\b", "reading"),
    (r"\brelniniscent\b", "reminiscent"),
    (r"\brheans\b", "means"),
    (r"\briticism\b", "criticism"),
    (r"\brnustn\b", "mustn"),
    (r"\bsexs\b", "sees"),
    (r"\bshor\b", "shop"),
    (r"\btnert\b", "inert"),
    (r"\btresome\b", "tiresome"),
    (r"\buitered\b", "uttered"),
    (r"\bwascharged\b", "was charged"),
    (r"\bwby\b", "why"),
    (r"\bweem\b", "seem"),
    (r"\bwichout\b", "without"),
    (r"\bwitk\b", "with"),
    (r"\bwoud\b", "would"),
    (r"\bwoun\b", "wound"),
    (r"\byealous\b", "jealous"),
    (r"\byoa\b", "you"),
    (r"\byouaren\b", "you aren"),
    (r"\byoull\b", "you’ll"),
    (r"\byourre\b", "you’re"),
    (r"\bretrain from\b", "refrain from"), (r"\bson\.ething\b", "something"), (r"\bf-ocks\b", "frocks"),
    # chapter openings are set in small capitals, which OCR reads as e.g. "Tue", "Matuieu", "It wAs"
    (r"\bMatu[ir]e?u\b", "Mathieu"), (r"\bTue\b", "The"), (r"\bWuart\b", "What"), (r"\bwAs\b", "was"),
    (r"\bDéme\b", "Dôme"), (r"\bJate\b", "late"), (r"\bnot seer to\b", "not seem to"),
    (r"^((?:A )?[A-Z])([A-Z]+)\b", lambda m: m.group(1) + m.group(2).lower()),   # SUMMER. -> Summer.
    (r"\s+([,.;:!?])", r"\1"),                               # space before punctuation
    (r"\.\s*\.\s*\.(?:\s*\.)?", "…"),                         # ". . ." -> …
    (r"\s*…\s*", "… "),
]


def resolve_input(arg):
    if arg:
        return Path(arg)
    pdfs = sorted(glob.glob(str(HERE / "raw" / "*.pdf")))
    if not pdfs:
        sys.exit("No PDF found. Pass the path explicitly or place it in ./raw/")
    return Path(pdfs[0])


def ocr_all(doc):
    OCR_DIR.mkdir(exist_ok=True)

    def one(i):
        tsv = OCR_DIR / f"p{i + 1:03d}.tsv"
        if tsv.exists() and tsv.stat().st_size:
            return
        png = OCR_DIR / f"p{i + 1:03d}.png"
        png.write_bytes(doc.extract_image(doc[i].get_images()[0][0])["image"])
        subprocess.run(["tesseract", str(png), str(tsv.with_suffix("")), "--dpi", "600", "-l", "eng",
                        "--psm", "4", "tsv"], check=True, capture_output=True,
                       env={**os.environ, "OMP_THREAD_LIMIT": "1"})
    with ThreadPoolExecutor(os.cpu_count() or 4) as ex:
        list(ex.map(one, range(len(doc))))


def page_lines(pno):
    """-> [(left, top, text)] for one page, page number removed."""
    lines = {}
    with open(OCR_DIR / f"p{pno:03d}.tsv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if r["level"] != "5" or not r["text"].strip():
                continue
            if float(r["conf"]) < 40 and not re.search(r"[A-Za-z0-9]", r["text"]):
                continue                                      # specks read as stray marks
            key = (r["block_num"], r["par_num"], r["line_num"])
            l, t = int(r["left"]), int(r["top"])
            if key not in lines:
                lines[key] = [l, t, []]
            lines[key][0] = min(lines[key][0], l)
            lines[key][2].append(r["text"])
    out = sorted(((l, t, " ".join(w)) for l, t, w in lines.values()), key=lambda x: x[1])
    if out and out[-1][1] > 3800 and len(out[-1][2]) <= 5:   # page number at the foot
        out.pop()
    return out


def strip_quotes(text):
    """Drop quotation marks; keep apostrophes inside words (don’t, Camus’s)."""
    text = re.sub(r"[“”\"„«»]", "", text)
    text = re.sub(r"(?<=[A-Za-z])[’'](?=[A-Za-z])", "\x02", text)   # protect in-word apostrophes
    text = re.sub(r"[‘’'`]", "", text)
    return text.replace("\x02", "’")


def resolve_wraps(text):
    def pick(m):
        left, right = m.group(1), m.group(2)
        lw, rw = re.sub(r"\W", "", left).lower(), re.sub(r"\W", "", right).lower()
        if lw + rw in _dict or lw not in _dict or rw not in _dict:
            return left + right
        return f"{left}-{right}"
    return re.sub(r"([\w’]+)" + WRAP + r"([\w’]+)", pick, text)


CONFUSIONS = [("c", "e"), ("e", "c"), ("l", "i"), ("i", "l"), ("rn", "m"), ("m", "rn"), ("v", "y"),
              ("ii", "u"), ("u", "ii"), ("n", "u"), ("u", "n"), ("li", "h"), ("h", "li"), ("cl", "d"),
              ("T", "I"), ("1", "l"), ("v", "u"), ("l", "f"), ("l", "h"), ("G", "C"), ("0", "o"), ("f", "t"), ("t", "f"), ("a", "o"), ("o", "a"),
              ("c", "o"), ("o", "c"), ("c", "d"), ("n", "h"), ("s", "e"), ("tl", "d"), ("I", "l"), ("l", "I"), ("", "")]
KEEP = {"des", "multi", "mio", "hafted", "caballo", "murrio", "aux", "ou"}   # French/Spanish/rare but right
NAMES: set = set()          # proper names seen often in the book (filled in main)
CHANGES: dict = {}          # every automatic word correction, for review (ocr_fixes.log)
BRITISH = [("our", "or"), ("ise", "ize"), ("is", "iz"), ("ys", "yz"), ("ence", "ense"), ("tre", "ter"),
           ("lled", "led"), ("lling", "ling"), ("ogue", "og"), ("aught", "aft"), ("ae", "e"), ("oe", "e")]


def known(w):
    return w.lower() in _dict or w in NAMES


def british(w):
    lw = w.lower()
    return any(a in lw and lw.replace(a, b) in _dict for a, b in BRITISH)


def fix_word(m):
    """Repair one OCR-misread word by trying the usual glyph confusions (c/e, l/i, rn/m ...).
    Lower-case words may become any dictionary word; capitalised words (names, French) only
    a name the book uses often (Mathicu -> Mathieu, Tvich -> Ivich)."""
    w = m.group(0)
    if len(w) < 3 or w.lower() in KEEP or known(w) or known(w.replace("’", "'")) or british(w):
        return w
    cap = w[0].isupper()
    best = []
    for a, b in CONFUSIONS:
        if not a:
            continue
        i = w.find(a)
        while i != -1:
            cand = w[:i] + b + w[i + len(a):]
            if (cand in NAMES) if cap else (cand.lower() in _dict and
                                            _dict.word_usage_frequency(cand.lower()) > (1e-5 if len(cand) <= 4 else 1e-8)):
                best.append(cand)
            i = w.find(a, i + 1)
    if not best:
        return w
    out = max(best, key=lambda c: (c in NAMES, _dict.word_usage_frequency(c.lower())))
    CHANGES[(w, out)] = CHANGES.get((w, out), 0) + 1
    return out


def finish(text):
    text = resolve_wraps(strip_quotes(text)).replace(WRAP, "")
    for pat, rep in FIXUPS:
        text = re.sub(pat, rep, text)
    text = re.sub(r"[^\W\d_][^\W_]*(?:’[a-z]+)?", fix_word, text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text if text.endswith(TERMINAL + (",",)) else text + "."


def main():
    src = resolve_input(sys.argv[1] if len(sys.argv) > 1 else None)
    doc = pymupdf.open(src)
    ocr_all(doc)
    chapters, paras, cur = [], None, ""

    def flush():
        nonlocal cur
        if cur and paras is not None:
            paras.append(cur)
        cur = ""

    for pno in range(FIRST_PAGE, LAST_PAGE + 1):
        lines = page_lines(pno)
        lefts = sorted(l for l, _, t in lines if len(t) > 30)
        margin = lefts[len(lefts) // 4] if lefts else 330       # lower quartile = body margin
        for left, top, text in lines:
            if text.strip() == "THE AGE OF REASON":
                continue
            m = CHAPTER.match(text.strip())
            if m:
                flush()
                paras = []
                chapters.append({"num": len(chapters) + 1, "label": f"Chapter {len(chapters) + 1}",
                                 "title": f"Chapter {len(chapters) + 1}", "start_page": pno,
                                 "paras": paras})
                continue
            if paras is None:
                continue
            if left > margin + INDENT_PX or not cur:
                flush()
                cur = text
            elif cur.endswith("-") and text[:1].islower():
                cur = cur[:-1] + WRAP + text
            else:
                cur = f"{cur} {text}"
    flush()

    # proper names: capitalised words the dictionary doesn't know that recur (Mathieu, Ivich, ...)
    from collections import Counter
    caps = Counter(w for ch in chapters for p in ch["paras"]
                   for w in re.findall(r"\b[A-Z][^\W\d_]+\b", p) if w.lower() not in _dict)
    NAMES.update(w for w, n in caps.items() if n >= 4 and w[0] not in "T"
                 or w in ("Brunet", "Daniel", "Boris", "Lola", "Sarah", "Jacques"))
    md = []
    for i, ch in enumerate(chapters):
        ch["end_page"] = chapters[i + 1]["start_page"] if i + 1 < len(chapters) else LAST_PAGE
        md.append(f"# {ch['title']}\n")
        md += [finish(p) + "\n" for p in ch.pop("paras") if re.search(r"[A-Za-z]{2}", p)]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    (HERE / "ocr_fixes.log").write_text("".join(f"{n:4d}  {a} -> {b}\n" for (a, b), n in
                                                sorted(CHANGES.items(), key=lambda kv: -kv[1])))
    (HERE / "chapters.json").write_text(json.dumps(chapters, indent=2), encoding="utf-8")
    print(f"{src.name}: {len(doc)} pages, {len(chapters)} chapters")
    print(f"wrote {OUT_MD.name} ({sum(len(m.split()) for m in md):,} words) and chapters.json")


if __name__ == "__main__":
    main()
