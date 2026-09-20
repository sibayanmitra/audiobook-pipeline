#!/usr/bin/env python3
"""
Audiobook pipeline: Titan — The Life of John D. Rockefeller, Sr. (Ron Chernow)
Multi-GPU data-parallel workers, batch-8 TTS, file-locked chapter queue.

Chapters come from chapters.json (built from the PDF's own TOC by pdf_to_md.py)
and are sliced out of Titan.md on its <!--PAGE:n--> markers.

Usage:
    taskset -c 0-15 nice -n 10 python audiobook_make.py
    taskset -c 0-15 nice -n 10 python audiobook_make.py --merge-only
    taskset -c 0-15 nice -n 10 python audiobook_make.py --test-chapter 1
    python audiobook_make.py --dry-run
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import static_ffmpeg
import torch

# ── config ────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).parent
MD_FILE      = HERE / "Titan.md"
CHAPTERS_JSON = HERE / "chapters.json"
OUT_DIR      = HERE / "audiobook"
CHAPTERS_DIR = OUT_DIR / "chapters"
FINAL_MP4    = OUT_DIR / "Titan.mp4"

MODEL_PATH   = HERE.parent / "models" / "Qwen3-TTS-12Hz-1.7B-CustomVoice"
SPEAKER      = "Ryan"
LANGUAGE     = "English"
SAMPLE_RATE  = 24000

GPUS         = [1, 2, 3]    # other users share these boxes; keep cuda:0 free
# A ~42 GB tenant sits on every card, leaving ~5.6 GiB. The model alone takes
# 4.18 GB, so a batch of 2 does not fit. Throughput is ~11 chars/s/GPU either
# way (batching buys nothing at this size), so batch 1 costs us almost nothing.
BATCH_SIZE   = 1
AAC_BITRATE  = "64k"        # mono narration; the volume is nearly full

OOM_RETRIES    = 20         # neighbour's memory use swings; wait it out
OOM_BACKOFF_S  = 15

PARA_SILENCE_S = 0.55
SENT_SILENCE_S = 0.15
MAX_SEG_CHARS  = 200        # shorter segments = lower peak VRAM, same chars/s
# ──────────────────────────────────────────────────────────────────────────────


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


# ── text cleaning ─────────────────────────────────────────────────────────────

_PAGE_MARK  = re.compile(r"<!--PAGE:\d+-->")
_PICTURE    = re.compile(r"\*{0,2}==>.*?intentionally omitted.*?<==\*{0,2}")
# photo credit lines: _**Caption.** (Courtesy of the Rockefeller Archive Center)_
_CAPTION    = re.compile(r"^_\*\*.*?\((?:Courtesy|Cou\w*)[^)]*\)_\s*$", re.M | re.S)
_CREDIT     = re.compile(r"\((?:Courtesy of|Photo by)[^)]*\)")
# endnote references left by the PDF: [ 1 ], [13 ], [204]
_ENDNOTE    = re.compile(r"\[\s*\d{1,3}\s*\]")
_MD_HEADER  = re.compile(r"^#{1,6}\s*", re.M)
_MD_BOLD    = re.compile(r"\*{1,2}|_{1,2}")
_MD_LINK    = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_CODE    = re.compile(r"`[^`]*`")
_MULTI_NL   = re.compile(r"\n{3,}")
_EM_DASH    = re.compile(r"[—–]")
_CURLY_APOS = re.compile("[‘’]")
_CURLY_QUOT = re.compile("[“”]")
_SPACES     = re.compile(r"[ \t]{2,}")


def clean(text: str) -> str:
    text = _PAGE_MARK.sub("", text)
    text = _CAPTION.sub("", text)
    text = _PICTURE.sub("", text)
    text = _CREDIT.sub("", text)
    text = _ENDNOTE.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_CODE.sub("", text)
    text = _MD_HEADER.sub("", text)
    text = _MD_BOLD.sub("", text)
    text = _EM_DASH.sub(", ", text)
    text = _CURLY_APOS.sub("'", text)
    text = _CURLY_QUOT.sub('"', text)
    text = unicodedata.normalize("NFKC", text)
    text = _SPACES.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


# ── segmentation ──────────────────────────────────────────────────────────────

_SENT_END = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text: str) -> list:
    return [s.strip() for s in _SENT_END.split(text) if s.strip()]


def hard_wrap(sent: str) -> list:
    """Break a monster sentence on commas/semicolons so no segment exceeds the cap."""
    if len(sent) <= MAX_SEG_CHARS:
        return [sent]
    out, buf = [], ""
    for piece in re.split(r'(?<=[,;:])\s+', sent):
        while len(piece) > MAX_SEG_CHARS:          # still too long: split on words
            cut = piece.rfind(" ", 0, MAX_SEG_CHARS) or MAX_SEG_CHARS
            out.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if len(buf) + len(piece) + 1 > MAX_SEG_CHARS and buf:
            out.append(buf.strip())
            buf = piece
        else:
            buf = (buf + " " + piece).strip() if buf else piece
    if buf:
        out.append(buf.strip())
    return [o for o in out if o]


def make_segments(paragraph: str) -> list:
    segments, buf = [], ""
    for sent in split_sentences(paragraph):
        for part in hard_wrap(sent):
            if len(buf) + len(part) + 1 > MAX_SEG_CHARS and buf:
                segments.append(buf.strip())
                buf = part
            else:
                buf = (buf + " " + part).strip() if buf else part
    if buf:
        segments.append(buf.strip())
    return segments


# ── chapter parsing ───────────────────────────────────────────────────────────

def load_pages() -> dict:
    md = MD_FILE.read_text(encoding="utf-8")
    parts = re.split(r"<!--PAGE:(\d+)-->", md)
    return {int(parts[i]): parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def parse_chapters() -> list:
    pages = load_pages()
    meta = json.loads(CHAPTERS_JSON.read_text(encoding="utf-8"))
    chapters = []
    for c in meta:
        body = "".join(
            pages.get(p, "") for p in range(c["start_page"], c["end_page"] + 1)
        )
        chapters.append({"num": c["num"], "title": c["title"], "text": clean(body)})
    return chapters


# ── chapter path helpers ──────────────────────────────────────────────────────

def chapter_paths(chapter: dict, order: int) -> Tuple[Path, Path, Path, Path]:
    slug = f"{order:03d}_{re.sub(r'[^a-z0-9]+', '_', chapter['title'].lower()).strip('_')}"
    return (
        CHAPTERS_DIR / f"{slug}.mp4",
        CHAPTERS_DIR / f"{slug}.raw",   # int16 PCM, streamed as we go
        CHAPTERS_DIR / f"{slug}.idx",   # resume checkpoint
        CHAPTERS_DIR / f"{slug}.lock",
    )


# ── TTS inference (batch) ─────────────────────────────────────────────────────

def load_model(device: str):
    from qwen_tts import Qwen3TTSModel
    print(f"[{device}] Loading model …", flush=True)
    model = Qwen3TTSModel.from_pretrained(
        str(MODEL_PATH), device_map=device, dtype=torch.bfloat16,
    )
    print(f"[{device}] Model ready.", flush=True)
    return model


def _generate(model, texts: list) -> list:
    wavs, _ = model.generate_custom_voice(
        text=texts,
        language=[LANGUAGE] * len(texts),
        speaker=[SPEAKER] * len(texts),
    )
    return [w.astype(np.float32) for w in wavs]


def synthesise_batch(model, texts: list, device: str) -> list:
    """Generate a batch, degrading to one-at-a-time when the card is squeezed.

    These GPUs are shared with a ~42 GB tenant, so free memory swings minute to
    minute. An OOM is a transient condition to wait out, not a reason to lose
    the chapter.
    """
    try:
        return _generate(model, texts)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(texts) > 1:
            print(f"  [{device}] OOM at batch {len(texts)} — splitting", flush=True)
            out = []
            for t in texts:
                out.extend(synthesise_batch(model, [t], device))
            return out

    # single segment still won't fit: wait for the neighbour to release memory
    for attempt in range(1, OOM_RETRIES + 1):
        time.sleep(OOM_BACKOFF_S * attempt)
        torch.cuda.empty_cache()
        try:
            return _generate(model, texts)
        except torch.cuda.OutOfMemoryError:
            print(f"  [{device}] OOM retry {attempt}/{OOM_RETRIES}", flush=True)
            torch.cuda.empty_cache()
    raise RuntimeError("persistent CUDA OOM on a single segment")


# ── chapter → WAV ─────────────────────────────────────────────────────────────

def to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def chapter_segments(chapter: dict) -> list:
    """Flat list of (text, ends_paragraph)."""
    flat = []
    for para in chapter["text"].split("\n\n"):
        para = para.strip()
        if not para:
            continue
        segs = make_segments(para)
        for j, seg in enumerate(segs):
            if seg:
                flat.append((seg, j == len(segs) - 1))
    return flat


def chapter_to_pcm(model, chapter: dict, raw_path: Path, idx_path: Path,
                   device: str) -> None:
    """Synthesise straight to an int16 PCM file, checkpointing after each segment.

    Chapters here run for hours on a contended box, so a killed run must resume
    where it stopped rather than start the chapter again.
    """
    flat = chapter_segments(chapter)
    total = len(flat)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    start = 0
    if idx_path.exists() and raw_path.exists():
        try:
            ck = json.loads(idx_path.read_text())
            if ck.get("total") == total:
                start = int(ck["segments"])
                # drop any bytes written after the last confirmed segment
                with open(raw_path, "r+b") as fh:
                    fh.truncate(int(ck["bytes"]))
                print(f"  [{device}] resuming at segment {start}/{total}", flush=True)
        except Exception:
            start = 0

    mode = "r+b" if start else "wb"
    t0 = time.time()
    with open(raw_path, mode) as fh:
        fh.seek(0, os.SEEK_END)
        for i in range(start, total, BATCH_SIZE):
            batch_items = flat[i: i + BATCH_SIZE]
            batch_texts = [t for t, _ in batch_items]
            batch_last  = [last for _, last in batch_items]

            audio_batch = synthesise_batch(model, batch_texts, device)
            # Hand the KV cache back every step; on a card this full, reserved
            # -but-unallocated blocks are what tip us into an OOM.
            torch.cuda.empty_cache()

            for audio, is_last in zip(audio_batch, batch_last):
                fh.write(to_pcm16(audio))
                fh.write(to_pcm16(silence(PARA_SILENCE_S if is_last else SENT_SILENCE_S)))

            fh.flush()
            os.fsync(fh.fileno())
            done = min(i + BATCH_SIZE, total)
            idx_path.write_text(json.dumps(
                {"segments": done, "bytes": fh.tell(), "total": total}))

            elapsed = time.time() - t0
            rate = (done - start) / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            if done % 10 == 0 or done == total:
                print(f"  [{device}] [{done}/{total}] {rate:.3f} seg/s | "
                      f"ETA {eta/60:.0f}m", flush=True)


# ── WAV → MP4 ─────────────────────────────────────────────────────────────────

def pcm_to_mp4(raw_path: Path, mp4_path: Path, idx_path: Path) -> None:
    static_ffmpeg.add_paths()
    ffmpeg = shutil.which("ffmpeg")
    cmd = [ffmpeg, "-y",
           "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", str(raw_path),
           "-c:a", "aac", "-b:a", AAC_BITRATE, str(mp4_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    raw_path.unlink()
    if idx_path.exists():
        idx_path.unlink()


# ── worker (one per GPU) ──────────────────────────────────────────────────────

def worker(gpu_id: int, chapters: list) -> None:
    device = f"cuda:{gpu_id}"
    model  = load_model(device)

    for order, chapter in chapters:
        mp4, raw, idx, lock = chapter_paths(chapter, order)

        if mp4.exists():
            print(f"  [{device}] SKIP {mp4.name}", flush=True)
            continue
        if lock.exists():
            print(f"  [{device}] LOCKED (other worker) {lock.name}", flush=True)
            continue
        try:
            lock.touch()
            print(f"\n  [{device}] {order:03d} {chapter['title']}", flush=True)
            chapter_to_pcm(model, chapter, raw, idx, device)
            pcm_to_mp4(raw, mp4, idx)
            print(f"  [{device}] -> {mp4.name}", flush=True)
        except Exception as e:
            # leave raw/idx in place so the next run resumes mid-chapter
            print(f"  [{device}] ERROR on {chapter['title']}: {e}", flush=True)
        finally:
            if lock.exists():
                lock.unlink()


def split_chapters(chapters: list, n: int) -> list:
    """Round-robin so each worker gets a mix of long and short chapters."""
    buckets = [[] for _ in range(n)]
    for i, ch in enumerate(chapters):
        buckets[i % n].append(ch)
    return buckets


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
    mp.set_start_method("spawn", force=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--merge-only",   action="store_true")
    ap.add_argument("--dry-run",      action="store_true",
                    help="Parse and segment only; print stats, no TTS.")
    ap.add_argument("--test-chapter", type=int, default=None,
                    help="Run a single chapter by order index (uses first GPU)")
    ap.add_argument("--batch", type=int, default=None,
                    help="Override BATCH_SIZE (GPU memory on this box is shared)")
    args = ap.parse_args()

    global BATCH_SIZE
    if args.batch:
        BATCH_SIZE = args.batch
        print(f"BATCH_SIZE = {BATCH_SIZE}")

    CHAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    for lf in CHAPTERS_DIR.glob("*.lock"):
        lf.unlink()

    if args.merge_only:
        merge_chapters()
        return

    chapters = list(enumerate(parse_chapters()))
    print(f"Parsed {len(chapters)} sections.")

    if args.dry_run:
        grand = 0
        for order, ch in chapters:
            segs = sum(len(make_segments(p)) for p in ch["text"].split("\n\n") if p.strip())
            grand += segs
            print(f"  {order:03d}  {segs:>5} segs  {len(ch['text']):>7,} chars  {ch['title']}")
        print(f"\nTotal {grand:,} segments.")
        chars = sum(len(c["text"]) for _, c in chapters)
        print(f"Total {chars:,} chars.")
        for n in (1, 2, 3):
            print(f"  ~{chars / (11.0 * n) / 3600:.1f} h wall-clock on {n} GPUs @11 chars/s/GPU")
        print(f"  ~{chars / 14.0 / 3600:.1f} h of finished audio"
              f"  |  ~{chars / 14.0 * 8000 / 1e9:.1f} GB at {AAC_BITRATE}")
        return

    if args.test_chapter is not None:
        worker(GPUS[0], [chapters[args.test_chapter]])
        return

    remaining = [(o, c) for o, c in chapters if not chapter_paths(c, o)[0].exists()]
    print(f"{len(chapters) - len(remaining)} already done, {len(remaining)} remaining.")

    if not remaining:
        print("All chapters done — running merge.")
        merge_chapters()
        return

    buckets = split_chapters(remaining, len(GPUS))
    processes = []
    for gpu_id, bucket in zip(GPUS, buckets):
        p = mp.Process(target=worker, args=(gpu_id, bucket), daemon=True)
        p.start()
        processes.append(p)
        print(f"Started worker on cuda:{gpu_id} with {len(bucket)} chapters.")

    for p in processes:
        p.join()

    print("\nAll workers done.")
    merge_chapters()


if __name__ == "__main__":
    main()
