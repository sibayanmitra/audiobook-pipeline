#!/usr/bin/env python3
"""Sanity checks for a book run through pdf_to_md.py + audiobook_make.py.

Usage:
    python check_audiobook.py --text     # after pdf_to_md.py: is the extracted text sane?
    python check_audiobook.py --audio    # after audiobook_make.py: is the audio sane?
    python check_audiobook.py            # both

Exit status is 1 if any FAIL was found (WARNs do not fail), so a queue can stop on it.
It imports paths and the chapter parser from this directory's audiobook_make.py, so each
book directory is checked against its own configuration.
"""
import argparse
import glob
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

import audiobook_make as am

HERE = Path(__file__).resolve().parent
results = []   # (level, check, detail)


def report(level: str, check: str, detail: str):
    results.append((level, check, detail))
    print(f"[{level:<4}] {check}: {detail}")


# ── text ──────────────────────────────────────────────────────────────────────

def check_text():
    if not am.MD_FILE.exists():
        report("FAIL", "markdown", f"{am.MD_FILE.name} missing - run pdf_to_md.py")
        return
    md = am.MD_FILE.read_text(encoding="utf-8")
    chapters = am.parse_chapters(md)
    words = {c["num"]: sum(len(t.split()) for k, t in c["items"] if k != "title") for c in chapters}
    total = sum(words.values())

    # 1. coverage vs the PDF's own text layer (headers/page numbers/footnotes explain a few %)
    pdfs = sorted(glob.glob(str(HERE / "raw" / "*.pdf")))
    if pdfs:
        raw = subprocess.run(["pdftotext", pdfs[0], "-"], capture_output=True, text=True).stdout
        ratio = total / max(1, len(raw.split()))
        lvl = "PASS" if 0.88 <= ratio <= 1.02 else ("WARN" if 0.75 <= ratio <= 1.1 else "FAIL")
        report(lvl, "coverage", f"{total:,} words vs {len(raw.split()):,} in the PDF text layer ({ratio:.1%})")
    else:
        report("WARN", "coverage", "no raw/*.pdf to compare against")

    # 2. chapters
    cj = HERE / "chapters.json"
    if cj.exists():
        expected = len(json.loads(cj.read_text()))
        report("PASS" if expected == len(chapters) else "FAIL", "chapters",
               f"{len(chapters)} in markdown, {expected} in chapters.json")
    tiny = [c["title"] for c in chapters if words[c["num"]] < 150]
    report("PASS" if not tiny else "WARN", "chapter size",
           "all chapters >=150 words" if not tiny else f"very short: {tiny}")

    # 3. leftovers that should never reach the speech engine
    body = "\n".join(t for c in chapters for k, t in c["items"] if k != "title")
    for name, pat in [("non-Latin letters (OCR)", r"[Ͱ-ӿ֐-ۿ]"),
                      ("stray * { } [ ] | ~ ^", r"[*{}\[\]|~^]"),
                      ("digit glued to word", r"[A-Za-z]\d\b|\b\d[A-Za-z]{2,}\b(?<!\dth)(?<!\dst)(?<!\dnd)(?<!\drd)"),
                      ("hyphen-wrap marker", "\x01"),
                      ("'Figure/Table N' captions", r"\b(?:Figure|Fig\.|Table) \d+")]:
        hits = re.findall(pat, body)
        report("PASS" if not hits else "WARN", name,
               "none" if not hits else f"{len(hits)} e.g. {sorted(set(hits))[:6]}")

    paras = [t for c in chapters for k, t in c["items"] if k == "para"]
    open_ended = [p[-50:] for p in paras if not p.rstrip().endswith(tuple(".!?:;\"')…"))]
    report("PASS" if len(open_ended) <= max(2, len(paras) // 200) else "WARN", "paragraph endings",
           f"{len(open_ended)}/{len(paras)} paragraphs do not end in punctuation"
           + (f", e.g. ...{open_ended[0]!r}" if open_ended else ""))

    # 4. spelling (info: OCR errors show up here)
    try:
        from spellchecker import SpellChecker
        sp = SpellChecker()
        unknown = sorted({w for w in re.findall(r"\b[a-z]{4,}\b", body) if w not in sp})
        report("PASS" if len(unknown) < 60 else "WARN", "unknown lowercase words",
               f"{len(unknown)}; first 25: {unknown[:25]}")
    except ImportError:
        report("WARN", "spelling", "pyspellchecker not installed")


# ── audio ─────────────────────────────────────────────────────────────────────

def decode(path: Path):
    """Yield float32 mono blocks (1 s each) decoded from an audio file."""
    proc = subprocess.Popen([am.find_tool("ffmpeg"), "-v", "error", "-i", str(path), "-f", "f32le",
                             "-ac", "1", "-ar", str(am.SAMPLE_RATE), "-"],
                            stdout=subprocess.PIPE)
    block = am.SAMPLE_RATE * 4
    while True:
        buf = proc.stdout.read(block)
        if not buf:
            break
        yield np.frombuffer(buf, dtype=np.float32)
    proc.wait()


def analyse(path: Path):
    """-> (peak, longest_silence_s, dead_seconds). Chapters are streamed, never loaded whole."""
    peak, run, longest, dead, clip = 0.0, 0, 0, 0, 0
    win = am.SAMPLE_RATE // 10          # 100 ms windows
    for block in decode(path):
        peak = max(peak, float(np.abs(block).max()))
        clip += int((np.abs(block) >= 0.999).sum())
        n = len(block) // win
        if n == 0:
            continue
        rms = np.sqrt((block[: n * win].reshape(n, win) ** 2).mean(axis=1))
        for quiet in rms < 10 ** (-55 / 20):          # below -55 dBFS = digital silence
            run = run + 1 if quiet else 0
            longest = max(longest, run)
            dead += int(quiet)
    return peak, longest / 10, dead / 10, clip


def check_audio():
    md = am.MD_FILE.read_text(encoding="utf-8") if am.MD_FILE.exists() else ""
    chapters = am.parse_chapters(md)
    missing = [c["title"] for c in chapters if not am.chapter_path(c, am.CHAPTERS_DIR).exists()]
    report("PASS" if not missing else "FAIL", "chapter files",
           f"{len(chapters) - len(missing)}/{len(chapters)} present" + (f"; missing {missing}" if missing else ""))

    rates, durs = {}, {}
    for c in chapters:
        p = am.chapter_path(c, am.CHAPTERS_DIR)
        if not p.exists():
            continue
        durs[c["num"]] = am.duration_s(p)
        w = sum(len(t.split()) for k, t in c["items"])
        rates[c["num"]] = durs[c["num"]] / max(1, w)           # seconds of audio per word
    if rates:
        med = statistics.median(rates.values())
        report("PASS", "speaking rate", f"median {60 / med:.0f} words/min")
        for c in chapters:
            if c["num"] in rates and not 0.75 <= rates[c["num"]] / med <= 1.25:
                report("FAIL", f"duration ch{c['num']:02d}",
                       f"{durs[c['num']] / 60:.1f} min is {rates[c['num']] / med:.0%} of the typical "
                       f"length for its word count - truncated or garbled? ({c['title']})")
    for c in chapters:
        p = am.chapter_path(c, am.CHAPTERS_DIR)
        if not p.exists():
            continue
        peak, longest, dead, clip = analyse(p)
        problems = []
        if peak < 0.05:
            problems.append(f"very quiet (peak {peak:.3f})")
        if clip > am.SAMPLE_RATE // 20:
            problems.append(f"clipping ({clip} samples)")
        if longest >= 4.0:
            problems.append(f"{longest:.1f}s of digital silence in a row")
        if dead / max(1, durs[c["num"]]) > 0.25:
            problems.append(f"{dead / durs[c['num']]:.0%} silence overall")
        report("FAIL" if problems else "PASS", f"audio ch{c['num']:02d}",
               "; ".join(problems) if problems else f"{durs[c['num']] / 60:.1f} min, peak {peak:.2f}, "
               f"longest silence {longest:.1f}s")

    if am.FINAL_M4B.exists():
        out = subprocess.run([am.find_tool("ffprobe"), "-v", "error",
                              "-show_entries", "chapter=id", "-of", "csv=p=0", str(am.FINAL_M4B)],
                             capture_output=True, text=True).stdout.split()
        total = am.duration_s(am.FINAL_M4B)
        diff = abs(total - sum(durs.values()))
        report("PASS" if len(out) == len(durs) else "FAIL", "m4b chapters",
               f"{len(out)} markers for {len(durs)} chapter files")
        report("PASS" if diff < 2 + len(durs) else "FAIL", "m4b duration",
               f"{total / 3600:.2f} h (chapters sum to {sum(durs.values()) / 3600:.2f} h)")
    else:
        report("WARN", "m4b", f"{am.FINAL_M4B.name} not built yet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="store_true")
    ap.add_argument("--audio", action="store_true")
    args = ap.parse_args()
    both = not (args.text or args.audio)
    if args.text or both:
        check_text()
    if args.audio or both:
        check_audio()
    fails = sum(1 for lvl, *_ in results if lvl == "FAIL")
    warns = sum(1 for lvl, *_ in results if lvl == "WARN")
    print(f"\n{len(results)} checks: {fails} FAIL, {warns} WARN")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
