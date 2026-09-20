#!/usr/bin/env python3
"""
Hackers & Painters audiobook pipeline — Qwen3-TTS, single GPU.

Usage:
    python audiobook_make.py --list-chapters   # sanity check parsing, no TTS
    python audiobook_make.py --test-chapter 1  # render one chapter only
    python audiobook_make.py                   # render everything + merge
    python audiobook_make.py --merge-only
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"   # single GPU only

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import static_ffmpeg
import torch

# ── config ────────────────────────────────────────────────────────────────────
MD_FILE      = Path(__file__).parent / "Hackers_and_Painters.md"
OUT_DIR      = Path(__file__).parent / "audiobook"
CHAPTERS_DIR = OUT_DIR / "chapters"
FINAL_MP4    = OUT_DIR / "Hackers_and_Painters.mp4"

MODEL_PATH   = Path(__file__).parent.parent / "models" / "Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEVICE       = "cuda:0"   # only cuda:2 is visible (CUDA_VISIBLE_DEVICES=2), so it's index 0 here
SPEAKER      = "Aiden"
LANGUAGE     = "English"
SAMPLE_RATE  = 24000
BATCH_SIZE   = 8

PARA_SILENCE_S = 0.55
SENT_SILENCE_S = 0.15
MAX_SEG_CHARS  = 280
# ──────────────────────────────────────────────────────────────────────────────


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


# ── text cleaning ─────────────────────────────────────────────────────────────

_IMAGE_RE  = re.compile(r"==>.*?intentionally omitted.*?<==", re.I)
_MD_HEADER = re.compile(r"^#{1,6}\s*", re.M)
_MD_BOLD   = re.compile(r"\*{1,2}|_{1,2}")
_MD_LINK   = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_CODE   = re.compile(r"`[^`]*`")
_MULTI_NL  = re.compile(r"\n{3,}")
_EM_DASH   = re.compile(r"—")
_CURLY_A   = re.compile("[‘’]")
_CURLY_Q   = re.compile("[“”]")


def clean(text: str) -> str:
    text = _IMAGE_RE.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_CODE.sub("", text)
    text = _MD_HEADER.sub("", text)
    text = _MD_BOLD.sub("", text)
    text = _EM_DASH.sub(", ", text)
    text = _CURLY_A.sub("'", text)
    text = _CURLY_Q.sub('"', text)
    text = unicodedata.normalize("NFKC", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


# ── chapter parsing ───────────────────────────────────────────────────────────
# This PDF's chapter-title headings render at large decorative font sizes and
# pymupdf4llm drops spaces inside them ("NerdsAre"). Each chapter also carries
# a small running-header repeat of its own title on every page, correctly
# spaced. So: group same-title heading variants together (compare with all
# non-alphanumeric characters stripped, which erases spacing differences),
# pick the best-spaced variant as the spoken title, and drop every variant
# from the body so it isn't repeated once per page.

_CHAPTER_MARK     = re.compile(r"^##\s*_Chapter\s+(\d+)_\s*$", re.M)
_NOTE_MARK        = re.compile(r"^##\s*_Note to readers_\s*$", re.M)
_CONTENTS_MARK    = re.compile(r"^##\s*Contents\s*$", re.M)
_PREFACE_MARK     = re.compile(r"^##\s*Preface\s*$", re.M)
_BACKMATTER_MARK  = re.compile(r"^##\s*Notes\s*$", re.M)
_HEADING_LINE     = re.compile(r"^#{1,6}[ \t]*(.+?)[ \t]*$")

# The book's own running-header footer ("Hackers & Painters") shows up on
# pages throughout, unrelated to which chapter it's on -> always noise.
_GLOBAL_FOOTER_KEYS = {"hackerspainters"}


def heading_key(raw: str) -> str:
    t = re.sub(r"[_*`#]", "", raw).lower()
    return re.sub(r"[^a-z0-9]", "", t)


def extract_title_and_body(span: str):
    lines = span.splitlines()
    heads = []
    for i, line in enumerate(lines):
        m = _HEADING_LINE.match(line)
        if m:
            heads.append((i, m.group(1)))

    groups = defaultdict(list)
    for i, raw in heads:
        key = heading_key(raw)
        if not key or key in _GLOBAL_FOOTER_KEYS:
            continue
        groups[key].append((i, raw))

    title = None
    title_idxs = set()
    if groups:
        _, occs = max(groups.items(), key=lambda kv: len(kv[1]))
        title = max((raw for _, raw in occs), key=lambda r: r.count(" "))
        title = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", title)
        title = re.sub(r"[_*]", "", title)
        title = re.sub(r"\s+", " ", title).strip()
        title_idxs = {i for i, _ in occs}

    drop = set(title_idxs)
    for i, raw in heads:
        key = heading_key(raw)
        if not key or key in _GLOBAL_FOOTER_KEYS:
            drop.add(i)

    body = "\n".join(ln for i, ln in enumerate(lines) if i not in drop)
    return title, body


def build_sections(md: str):
    chapter_matches = list(_CHAPTER_MARK.finditer(md))
    note_m       = _NOTE_MARK.search(md)
    contents_m   = _CONTENTS_MARK.search(md)
    preface_m    = _PREFACE_MARK.search(md)
    backmatter_m = _BACKMATTER_MARK.search(md)
    end_of_book  = backmatter_m.start() if backmatter_m else len(md)

    sections = []  # (num, label, raw_span)

    if note_m and contents_m and preface_m:
        note_span    = md[note_m.end():contents_m.start()]
        preface_end  = chapter_matches[0].start() if chapter_matches else end_of_book
        preface_span = md[preface_m.end():preface_end]
        sections.append((0, "Preface", note_span + "\n\n" + preface_span))

    for idx, m in enumerate(chapter_matches):
        num   = int(m.group(1))
        start = m.end()
        end   = chapter_matches[idx + 1].start() if idx + 1 < len(chapter_matches) else end_of_book
        sections.append((num, f"Chapter {num}", md[start:end]))

    return sections


def parse_chapters(md: str) -> list:
    chapters = []
    for num, label, span in build_sections(md):
        title, body = extract_title_and_body(span)
        if not title:
            title = label
        header = f"{label}. {title}." if num > 0 else f"{title}."
        text = clean(header + "\n\n" + body)
        chapters.append({"num": num, "title": title, "text": text})
    return chapters


# ── segmentation ──────────────────────────────────────────────────────────────

_SENT_END = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text: str) -> list:
    return [s.strip() for s in _SENT_END.split(text) if s.strip()]


def make_segments(paragraph: str) -> list:
    sentences = split_sentences(paragraph)
    segments, buf = [], ""
    for sent in sentences:
        if not sent:
            continue
        if len(buf) + len(sent) + 1 > MAX_SEG_CHARS and buf:
            segments.append(buf.strip())
            buf = sent
        else:
            buf = (buf + " " + sent).strip() if buf else sent
    if buf:
        segments.append(buf.strip())
    return segments


# ── helpers ───────────────────────────────────────────────────────────────────

def chapter_slug(chapter: dict) -> str:
    return f"{chapter['num']:03d}_{re.sub(r'[^a-z0-9]+', '_', chapter['title'].lower())[:40]}"


def chapter_mp4(chapter: dict) -> Path:
    return CHAPTERS_DIR / f"{chapter_slug(chapter)}.mp4"


def wav_to_mp4(wav_path: Path, mp4_path: Path) -> None:
    static_ffmpeg.add_paths()
    ffmpeg = shutil.which("ffmpeg")
    cmd = [ffmpeg, "-y", "-i", str(wav_path),
           "-c:a", "aac", "-b:a", "128k", str(mp4_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wav_path.unlink()


# ── TTS inference (batch, single GPU) ─────────────────────────────────────────

def load_model():
    from qwen_tts import Qwen3TTSModel
    print(f"[{DEVICE}] Loading Qwen3-TTS model …", flush=True)
    model = Qwen3TTSModel.from_pretrained(
        str(MODEL_PATH), device_map=DEVICE, dtype=torch.bfloat16,
    )
    print(f"[{DEVICE}] Model ready.", flush=True)
    return model


def synthesise_batch(model, texts: list) -> list:
    wavs, _ = model.generate_custom_voice(
        text=texts,
        language=[LANGUAGE] * len(texts),
        speaker=[SPEAKER] * len(texts),
    )
    return [w.astype(np.float32) for w in wavs]


def chapter_to_wav(model, chapter: dict, wav_path: Path) -> None:
    paragraphs = [p.strip() for p in chapter["text"].split("\n\n") if p.strip()]

    flat = []
    for para in paragraphs:
        segs = make_segments(para)
        for j, seg in enumerate(segs):
            if seg:
                flat.append((seg, j == len(segs) - 1))

    total = len(flat)
    pieces = []
    t0 = time.time()

    for i in range(0, total, BATCH_SIZE):
        batch_items = flat[i: i + BATCH_SIZE]
        batch_texts = [t for t, _ in batch_items]
        batch_last  = [last for _, last in batch_items]

        audio_batch = synthesise_batch(model, batch_texts)

        for audio, is_last in zip(audio_batch, batch_last):
            pieces.append(audio)
            pieces.append(silence(PARA_SILENCE_S if is_last else SENT_SILENCE_S))

        done = min(i + BATCH_SIZE, total)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  [{done}/{total}] {rate:.2f} seg/s | ETA {eta/60:.1f}m", flush=True)

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_path), np.concatenate(pieces), SAMPLE_RATE)


# ── merge ─────────────────────────────────────────────────────────────────────

def merge_chapters() -> None:
    static_ffmpeg.add_paths()
    ffmpeg = shutil.which("ffmpeg")
    mp4s = sorted(CHAPTERS_DIR.glob("*.mp4"))
    if not mp4s:
        sys.exit("No chapter MP4s found.")

    FINAL_MP4.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as flist:
        for p in mp4s:
            flist.write(f"file '{p.resolve()}'\n")
        flist_path = flist.name

    print(f"\nMerging {len(mp4s)} chapters -> {FINAL_MP4}")
    cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0",
           "-i", flist_path, "-c", "copy", str(FINAL_MP4)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    os.unlink(flist_path)
    print(f"Done -> {FINAL_MP4}  ({FINAL_MP4.stat().st_size/1e6:.0f} MB)")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge-only",    action="store_true")
    ap.add_argument("--test-chapter",  type=int, default=None,
                    help="Render a single chapter number for testing")
    ap.add_argument("--list-chapters", action="store_true")
    args = ap.parse_args()

    CHAPTERS_DIR.mkdir(parents=True, exist_ok=True)

    md       = MD_FILE.read_text(encoding="utf-8")
    chapters = parse_chapters(md)

    if args.list_chapters:
        print(f"Parsed {len(chapters)} sections.\n")
        for ch in chapters:
            words = len(ch["text"].split())
            print(f"  [{ch['num']:>2}] {ch['title']!r:45s} ~{words:,} words")
        return

    if args.merge_only:
        merge_chapters()
        return

    print(f"Parsed {len(chapters)} sections.")

    if args.test_chapter is not None:
        chapters = [c for c in chapters if c["num"] == args.test_chapter]
        if not chapters:
            sys.exit(f"No chapter numbered {args.test_chapter}")

    remaining = [ch for ch in chapters if not chapter_mp4(ch).exists()]
    print(f"{len(chapters) - len(remaining)} already done, {len(remaining)} remaining.")

    if not remaining:
        if args.test_chapter is None:
            merge_chapters()
        return

    model = load_model()

    for i, chapter in enumerate(remaining):
        mp4 = chapter_mp4(chapter)
        wav = mp4.with_suffix(".wav")
        print(f"\n[{i+1}/{len(remaining)}] Chapter {chapter['num']}: {chapter['title']}", flush=True)
        try:
            chapter_to_wav(model, chapter, wav)
            wav_to_mp4(wav, mp4)
            print(f"  -> {mp4.name}", flush=True)
        except Exception as e:
            print(f"  ERROR on chapter {chapter['num']}: {e}", flush=True)
            if wav.exists():
                wav.unlink()

    if args.test_chapter is None:
        merge_chapters()


if __name__ == "__main__":
    main()
