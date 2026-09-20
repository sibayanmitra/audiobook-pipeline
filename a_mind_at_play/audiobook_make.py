#!/usr/bin/env python3
"""
A Mind at Play — Kokoro audiobook pipeline, 4-GPU parallel.

Usage:
    python audiobook_make.py
    python audiobook_make.py --merge-only
    python audiobook_make.py --test-chapter 0
    python audiobook_make.py --gpus 0 1          # use only GPUs 0 and 1
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

import numpy as np
import soundfile as sf
import static_ffmpeg

# ── config ────────────────────────────────────────────────────────────────────
MD_FILE      = Path(__file__).parent / "a_mind_at_play.md"
OUT_DIR      = Path(__file__).parent / "audiobook"
CHAPTERS_DIR = OUT_DIR / "chapters"
FINAL_MP4    = OUT_DIR / "A_Mind_at_Play.mp4"

MODEL_PATH   = "/home/sibayan_mitra_2024/audio/models/kokoro-onnx/kokoro-v1.0.fp16.onnx"
VOICES_PATH  = "/home/sibayan_mitra_2024/audio/models/kokoro-onnx/voices-v1.0.bin"

VOICE        = "am_echo"
SPEED        = 1.0
LANG         = "en-us"
SAMPLE_RATE  = 24000

DEFAULT_GPUS   = [0]
PARA_SILENCE_S = 0.50
SENT_SILENCE_S = 0.12
MAX_SEG_CHARS  = 180    # well under Kokoro's 510-phoneme limit
# ──────────────────────────────────────────────────────────────────────────────

# CUDA libs bundled with qwen-tts-venv (needed by onnxruntime-gpu).
# Must be set at module level so spawned processes inherit before dlopen.
_NVIDIA = "/home/sibayan_mitra_2024/audio/qwen-tts-venv/lib/python3.10/site-packages/nvidia"
_EXTRA_LIB = f"{_NVIDIA}/cudnn/lib:{_NVIDIA}/cuda_runtime/lib"
os.environ["LD_LIBRARY_PATH"] = _EXTRA_LIB + ":" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["ONNX_PROVIDER"]   = "CUDAExecutionProvider"


def silence(seconds):
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


# ── text cleaning ─────────────────────────────────────────────────────────────

_IMAGE_RE  = re.compile(r"==>.*?intentionally omitted.*?<==", re.I)
_MD_HEADER = re.compile(r"^#{1,6}\s*", re.M)
_MD_BOLD   = re.compile(r"\*{1,2}|_{1,2}")
_MD_LINK   = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_CODE   = re.compile(r"`[^`]*`")
_MULTI_NL  = re.compile(r"\n{3,}")
_EM_DASH   = re.compile(r"—")
_CURLY_A   = re.compile(u"[‘’]")
_CURLY_Q   = re.compile(u"[“”]")


def clean(text):
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


# ── segmentation ──────────────────────────────────────────────────────────────

_SENT_END = re.compile(r'(?<=[.!?])\s+')


def make_segments(paragraph):
    sentences = [s.strip() for s in _SENT_END.split(paragraph) if s.strip()]
    segments, buf = [], ""
    for sent in sentences:
        if len(buf) + len(sent) + 1 > MAX_SEG_CHARS and buf:
            segments.append(buf.strip())
            buf = sent
        else:
            buf = (buf + " " + sent).strip() if buf else sent
    if buf:
        segments.append(buf.strip())
    return segments


# ── chapter parsing ───────────────────────────────────────────────────────────

# Only match bold chapter headers (## **Title**); skips quote/equation/lyric ## lines
_BOLD_HEADER  = re.compile(r'^## \*\*(.+?)\*\*\s*$', re.M)
_PURE_NUMBER  = re.compile(r'^\d+$')
_BACK_MATTER  = {'Photographs', 'Acknowledgments', 'About the Authors',
                 'Notes', 'Bibliography', 'Index'}
_BOILERPLATE  = {'Contents', 'CLICK HERE TO SIGN UP',
                 'Thank you for downloading this Simon & Schuster ebook.'}


_STRIP_TITLE = re.compile(u'[*_`#""''"]')


def _is_real_chapter(title):
    t = _STRIP_TITLE.sub('', title).strip()
    if t.startswith(u'—') or t.startswith('-'):
        return False
    if _PURE_NUMBER.match(t):
        return False
    if t in _BACK_MATTER or t in _BOILERPLATE:
        return False
    return True


def parse_chapters(md):
    intro  = re.search(r'^## \*\*INTRODUCTION\*\*', md, re.M)
    photos = re.search(r'^## \*\*Photographs\*\*',  md, re.M)
    if intro:
        md = md[intro.start():]
    if photos:
        cut = re.search(r'^## \*\*Photographs\*\*', md, re.M)
        if cut:
            md = md[:cut.start()]

    all_matches = list(_BOLD_HEADER.finditer(md))
    if not all_matches:
        return [{'num': 0, 'title': 'A Mind at Play', 'text': clean(md)}]

    real = [m for m in all_matches if _is_real_chapter(m.group(1))]

    raw = []
    for i, m in enumerate(real):
        title = _STRIP_TITLE.sub('', m.group(1)).strip()
        start = m.start()
        end   = real[i + 1].start() if i + 1 < len(real) else len(md)
        body  = clean(md[start:end])
        raw.append({'num': i, 'title': title, 'text': body})

    return raw


# ── helpers ───────────────────────────────────────────────────────────────────

def chapter_slug(chapter):
    return f"{chapter['num']:03d}_{re.sub(r'[^a-z0-9]+', '_', chapter['title'].lower())[:40]}"


def chapter_paths(chapter):
    slug = chapter_slug(chapter)
    return (
        CHAPTERS_DIR / f"{slug}.mp4",
        CHAPTERS_DIR / f"{slug}.wav",
        CHAPTERS_DIR / f"{slug}.lock",
    )


def wav_to_mp4(wav_path, mp4_path):
    static_ffmpeg.add_paths()
    ffmpeg = shutil.which("ffmpeg")
    subprocess.run(
        [ffmpeg, "-y", "-i", str(wav_path), "-c:a", "aac", "-b:a", "128k", str(mp4_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    wav_path.unlink()


# ── synthesis ─────────────────────────────────────────────────────────────────

def chapter_to_mp4(kokoro, chapter, mp4_path, label=""):
    paragraphs = [p.strip() for p in chapter["text"].split("\n\n") if p.strip()]
    flat = []
    for para in paragraphs:
        for j, seg in enumerate(make_segments(para)):
            if seg:
                flat.append((seg, j == len(make_segments(para)) - 1))

    total, pieces, t0 = len(flat), [], time.time()

    for i, (seg, is_last) in enumerate(flat):
        try:
            audio, _ = kokoro.create(seg, voice=VOICE, speed=SPEED, lang=LANG)
        except Exception:
            mid = len(seg) // 2
            split_at = seg.rfind(" ", 0, mid) or mid
            parts = []
            for part in [seg[:split_at].strip(), seg[split_at:].strip()]:
                if part:
                    a, _ = kokoro.create(part, voice=VOICE, speed=SPEED, lang=LANG)
                    parts.append(a.astype(np.float32))
            audio = np.concatenate(parts) if parts else np.zeros(SAMPLE_RATE, dtype=np.float32)

        pieces.append(audio.astype(np.float32))
        pieces.append(silence(PARA_SILENCE_S if is_last else SENT_SILENCE_S))
        elapsed = time.time() - t0
        rate    = (i + 1) / elapsed if elapsed else 0
        eta     = (total - i - 1) / rate if rate else 0
        print(f"  {label}[{i+1}/{total}] {rate:.2f} seg/s | ETA {eta/60:.1f}m", end="\r", flush=True)

    print()
    wav = mp4_path.with_suffix(".wav")
    sf.write(str(wav), np.concatenate(pieces), SAMPLE_RATE)
    wav_to_mp4(wav, mp4_path)


# ── worker (one per GPU) ──────────────────────────────────────────────────────

def worker(gpu_id, chapters):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from kokoro_onnx import Kokoro
    label = f"[GPU{gpu_id}] "
    print(f"{label}Loading Kokoro …", flush=True)
    kokoro = Kokoro(MODEL_PATH, VOICES_PATH)
    print(f"{label}Ready.", flush=True)

    for chapter in chapters:
        mp4, wav, lock = chapter_paths(chapter)

        if mp4.exists():
            print(f"{label}SKIP {mp4.name}", flush=True)
            continue
        if lock.exists():
            print(f"{label}LOCKED {lock.name}", flush=True)
            continue

        try:
            lock.touch()
            print(f"\n{label}Chapter {chapter['num']}: {chapter['title']}", flush=True)
            chapter_to_mp4(kokoro, chapter, mp4, label=label)
            print(f"{label}-> {mp4.name}", flush=True)
        except Exception as e:
            print(f"{label}ERROR on ch{chapter['num']}: {e}", flush=True)
            if wav.exists():
                wav.unlink()
        finally:
            if lock.exists():
                lock.unlink()


# ── merge ─────────────────────────────────────────────────────────────────────

def merge_chapters():
    static_ffmpeg.add_paths()
    ffmpeg = shutil.which("ffmpeg")
    mp4s   = sorted(CHAPTERS_DIR.glob("*.mp4"))
    if not mp4s:
        sys.exit("No chapter MP4s found.")
    FINAL_MP4.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in mp4s:
            f.write(f"file '{p.resolve()}'\n")
        flist = f.name
    print(f"\nMerging {len(mp4s)} chapters -> {FINAL_MP4}")
    subprocess.run(
        [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", flist, "-c", "copy", str(FINAL_MP4)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    os.unlink(flist)
    print(f"Done -> {FINAL_MP4}  ({FINAL_MP4.stat().st_size/1e6:.0f} MB)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    mp.set_start_method("spawn", force=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--merge-only",   action="store_true")
    ap.add_argument("--test-chapter", type=int, default=None)
    ap.add_argument("--gpus", type=int, nargs="+", default=DEFAULT_GPUS,
                    help="GPU IDs to use (default: 0 1 2 3)")
    args = ap.parse_args()

    CHAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    for lf in CHAPTERS_DIR.glob("*.lock"):
        lf.unlink()

    if args.merge_only:
        merge_chapters()
        return

    md       = MD_FILE.read_text(encoding="utf-8")
    chapters = parse_chapters(md)
    print(f"Parsed {len(chapters)} chapters.")

    if args.test_chapter is not None:
        worker(args.gpus[0], [chapters[args.test_chapter]])
        return

    remaining = [ch for ch in chapters if not chapter_paths(ch)[0].exists()]
    print(f"{len(chapters)-len(remaining)} done, {len(remaining)} remaining.")

    if not remaining:
        merge_chapters()
        return

    # Round-robin split across GPUs
    n = len(args.gpus)
    buckets = [[] for _ in range(n)]
    for i, ch in enumerate(remaining):
        buckets[i % n].append(ch)

    processes = []
    for gpu_id, bucket in zip(args.gpus, buckets):
        p = mp.Process(target=worker, args=(gpu_id, bucket), daemon=True)
        p.start()
        processes.append(p)
        print(f"Started worker on GPU {gpu_id} with {len(bucket)} chapters.")

    for p in processes:
        p.join()

    print("\nAll workers done.")
    merge_chapters()


if __name__ == "__main__":
    main()
