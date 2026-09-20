#!/usr/bin/env python3
"""
Audiobook pipeline: One Up On Wall Street
2-GPU data-parallel workers, batch-8 TTS, file-locked chapter queue.

Usage:
    taskset -c 0-15 nice -n 10 python audiobook_make.py
    taskset -c 0-15 nice -n 10 python audiobook_make.py --merge-only
    taskset -c 0-15 nice -n 10 python audiobook_make.py --test-chapter 1
"""

import argparse
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
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import static_ffmpeg
import torch

# ── config ────────────────────────────────────────────────────────────────────
MD_FILE      = Path(__file__).parent / "One_Up_On_WallStreet.md"
OUT_DIR      = Path(__file__).parent / "audiobook"
CHAPTERS_DIR = OUT_DIR / "chapters"
FINAL_MP4    = OUT_DIR / "One_Up_On_WallStreet.mp4"

MODEL_PATH   = Path(__file__).parent.parent / "models" / "Qwen3-TTS-12Hz-1.7B-CustomVoice"
SPEAKER      = "Ryan"
LANGUAGE     = "English"
SAMPLE_RATE  = 24000

GPUS         = [2, 3]       # 2 GPUs (2 & 3)
BATCH_SIZE   = 8            # segments per GPU call

PARA_SILENCE_S = 0.55
SENT_SILENCE_S = 0.15
MAX_SEG_CHARS  = 280
# ──────────────────────────────────────────────────────────────────────────────


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


# ── text cleaning ─────────────────────────────────────────────────────────────

_IMAGE_RE   = re.compile(r"==>.*?intentionally omitted.*?<==", re.I)
_MD_HEADER  = re.compile(r"^#{1,6}\s*", re.M)
_MD_BOLD    = re.compile(r"\*{1,2}|_{1,2}")
_MD_LINK    = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_CODE    = re.compile(r"`[^`]*`")
_MULTI_NL   = re.compile(r"\n{3,}")
_EM_DASH    = re.compile(r"—")
_CURLY_APOS = re.compile("[‘’]")
_CURLY_QUOT = re.compile("[“”]")


def clean(text: str) -> str:
    text = _IMAGE_RE.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_CODE.sub("", text)
    text = _MD_HEADER.sub("", text)
    text = _MD_BOLD.sub("", text)
    text = _EM_DASH.sub(", ", text)
    text = _CURLY_APOS.sub("'", text)
    text = _CURLY_QUOT.sub('"', text)
    text = unicodedata.normalize("NFKC", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


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


# ── chapter parsing ───────────────────────────────────────────────────────────

_CHAPTER_HEADER = re.compile(r"^##\s+\*\*(\d{1,2})\s+(.+?)\*\*\s*$", re.M)


def parse_chapters(md: str) -> list:
    matches = list(_CHAPTER_HEADER.finditer(md))
    if not matches:
        return [{"num": 0, "title": "One Up On Wall Street", "text": clean(md)}]

    raw = []
    front = md[: matches[0].start()].strip()
    if front:
        raw.append({"num": 0, "title": "Introduction", "text": clean(front), "_pos": 0})

    for i, m in enumerate(matches):
        num   = int(m.group(1))
        title = re.sub(r"[*_`#]", "", m.group(2)).strip()
        start = m.start()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        raw.append({"num": num, "title": title, "text": clean(md[start:end]), "_pos": start})

    seen = {}
    for ch in raw:
        n = ch["num"]
        if n not in seen or len(ch["text"].split()) > len(seen[n]["text"].split()):
            seen[n] = ch

    chapters = sorted(
        (c for c in seen.values() if c["num"] <= 200),
        key=lambda c: c["_pos"]
    )
    for ch in chapters:
        del ch["_pos"]
    return chapters


# ── chapter path helpers ──────────────────────────────────────────────────────

def chapter_paths(chapter: dict) -> Tuple[Path, Path, Path]:
    slug = f"{chapter['num']:03d}_{re.sub(r'[^a-z0-9]+', '_', chapter['title'].lower())}"
    mp4  = CHAPTERS_DIR / f"{slug}.mp4"
    wav  = CHAPTERS_DIR / f"{slug}.wav"
    lock = CHAPTERS_DIR / f"{slug}.lock"
    return mp4, wav, lock


# ── TTS inference (batch) ─────────────────────────────────────────────────────

def load_model(device: str):
    from qwen_tts import Qwen3TTSModel
    print(f"[{device}] Loading model …", flush=True)
    model = Qwen3TTSModel.from_pretrained(
        str(MODEL_PATH), device_map=device, dtype=torch.bfloat16,
    )
    print(f"[{device}] Model ready.", flush=True)
    return model


def synthesise_batch(model, texts: list) -> list:
    """Run a batch of text segments through TTS, return list of np.ndarray wavs."""
    wavs, _ = model.generate_custom_voice(
        text=texts,
        language=[LANGUAGE] * len(texts),
        speaker=[SPEAKER] * len(texts),
    )
    return [w.astype(np.float32) for w in wavs]


# ── chapter → WAV ─────────────────────────────────────────────────────────────

def chapter_to_wav(model, chapter: dict, wav_path: Path, device: str) -> None:
    paragraphs = [p.strip() for p in chapter["text"].split("\n\n") if p.strip()]

    # Build flat list of (seg_text, is_last_in_para) for silence logic
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
            if is_last:
                pieces.append(silence(PARA_SILENCE_S))
            else:
                pieces.append(silence(SENT_SILENCE_S))

        done = min(i + BATCH_SIZE, total)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  [{device}] [{done}/{total}] {rate:.2f} seg/s | ETA {eta/60:.1f}m", flush=True)

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_path), np.concatenate(pieces), SAMPLE_RATE)


# ── WAV → MP4 ─────────────────────────────────────────────────────────────────

def wav_to_mp4(wav_path: Path, mp4_path: Path) -> None:
    static_ffmpeg.add_paths()
    ffmpeg = shutil.which("ffmpeg")
    cmd = [ffmpeg, "-y", "-i", str(wav_path),
           "-c:a", "aac", "-b:a", "128k", str(mp4_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wav_path.unlink()


# ── worker (one per GPU) ──────────────────────────────────────────────────────

def worker(gpu_id: int, chapters: list) -> None:
    device = f"cuda:{gpu_id}"
    model  = load_model(device)

    for chapter in chapters:
        mp4, wav, lock = chapter_paths(chapter)

        # Skip if already done
        if mp4.exists():
            print(f"  [{device}] SKIP {mp4.name}", flush=True)
            continue

        # Claim chapter with a lock file
        if lock.exists():
            print(f"  [{device}] LOCKED (other worker) {lock.name}", flush=True)
            continue
        try:
            lock.touch()
            print(f"\n  [{device}] Chapter {chapter['num']}: {chapter['title']}", flush=True)
            chapter_to_wav(model, chapter, wav, device)
            wav_to_mp4(wav, mp4)
            print(f"  [{device}] -> {mp4.name}", flush=True)
        except Exception as e:
            print(f"  [{device}] ERROR on chapter {chapter['num']}: {e}", flush=True)
            if wav.exists():
                wav.unlink()
        finally:
            if lock.exists():
                lock.unlink()


# ── split chapters across 2 workers ──────────────────────────────────────────

def split_chapters(chapters: list, n: int) -> list:
    """Round-robin split so both workers get a mix of long and short chapters."""
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
    ap.add_argument("--merge-only",    action="store_true")
    ap.add_argument("--test-chapter",  type=int, default=None,
                    help="Run a single chapter index for testing (uses cuda:0)")
    args = ap.parse_args()

    CHAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    # clean up any stale lock files from a previous interrupted run
    for lf in CHAPTERS_DIR.glob("*.lock"):
        lf.unlink()

    if args.merge_only:
        merge_chapters()
        return

    md       = MD_FILE.read_text(encoding="utf-8")
    chapters = parse_chapters(md)
    print(f"Parsed {len(chapters)} chapters.")

    if args.test_chapter is not None:
        ch = chapters[args.test_chapter]
        worker(0, [ch])
        return

    # Filter to only chapters not yet done
    remaining = [ch for ch in chapters if not chapter_paths(ch)[0].exists()]
    print(f"{len(chapters) - len(remaining)} already done, {len(remaining)} remaining.")

    if not remaining:
        print("All chapters done — running merge.")
        merge_chapters()
        return

    # Split round-robin across 2 GPUs
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
