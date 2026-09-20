#!/usr/bin/env python3
"""
Personal MBA audiobook pipeline — Kokoro on GPU 2
Usage:
    python audiobook_make.py
    python audiobook_make.py --merge-only
    python audiobook_make.py --test-chapter 1
"""
import os
# Point onnxruntime-gpu to the cuDNN/CUDA libs bundled in the qwen-tts venv
_NVIDIA = "/home/sibayan_mitra_2024/audio/qwen-tts-venv/lib/python3.10/site-packages/nvidia"
os.environ["LD_LIBRARY_PATH"] = (
    f"{_NVIDIA}/cudnn/lib:{_NVIDIA}/cuda_runtime/lib:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["ONNX_PROVIDER"]        = "CUDAExecutionProvider"

import argparse
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
from kokoro_onnx import Kokoro

# ── config ────────────────────────────────────────────────────────────────────
MD_FILE      = Path(__file__).parent / "the_personal_mba.md"
OUT_DIR      = Path(__file__).parent / "audiobook"
CHAPTERS_DIR = OUT_DIR / "chapters"
FINAL_MP4    = OUT_DIR / "The_Personal_MBA.mp4"
MODEL_PATH   = "/home/sibayan_mitra_2024/audio/models/kokoro-onnx/kokoro-v1.0.fp16.onnx"
VOICES_PATH  = "/home/sibayan_mitra_2024/audio/models/kokoro-onnx/voices-v1.0.bin"

VOICE        = "am_echo"
SPEED        = 1.0
LANG         = "en-us"
SAMPLE_RATE  = 24000

PARA_SILENCE_S = 0.50
SENT_SILENCE_S = 0.12
MAX_SEG_CHARS  = 180    # keep well under Kokoro's 510-phoneme limit
# ──────────────────────────────────────────────────────────────────────────────


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
_CURLY_A   = re.compile("[‘’]")
_CURLY_Q   = re.compile("[“”]")


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

_CHAPTER_HEADER = re.compile(r"^#{1,3}\s+(.+)$", re.M)


def parse_chapters(md):
    matches = list(_CHAPTER_HEADER.finditer(md))
    if not matches:
        return [{"num": 0, "title": "The Personal MBA", "text": clean(md)}]

    raw = []
    front = md[: matches[0].start()].strip()
    if front:
        raw.append({"num": 0, "title": "Introduction", "text": clean(front), "_pos": 0})

    for i, m in enumerate(matches):
        title = re.sub(r"[*_`#]", "", m.group(1)).strip()
        if not title or len(title) > 80:
            continue
        start = m.start()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        body  = clean(md[start:end])
        if len(body.split()) < 30:   # skip near-empty stubs
            continue
        raw.append({"num": len(raw), "title": title, "text": body, "_pos": start})

    chapters = sorted(raw, key=lambda c: c["_pos"])
    for ch in chapters:
        del ch["_pos"]
    # renumber sequentially
    for i, ch in enumerate(chapters):
        ch["num"] = i
    return chapters


# ── helpers ───────────────────────────────────────────────────────────────────

def chapter_slug(chapter):
    return f"{chapter['num']:03d}_{re.sub(r'[^a-z0-9]+', '_', chapter['title'].lower())[:40]}"


def chapter_mp4(chapter):
    return CHAPTERS_DIR / f"{chapter_slug(chapter)}.mp4"


def wav_to_mp4(wav_path, mp4_path):
    static_ffmpeg.add_paths()
    ffmpeg = shutil.which("ffmpeg")
    subprocess.run([ffmpeg, "-y", "-i", str(wav_path),
                    "-c:a", "aac", "-b:a", "128k", str(mp4_path)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wav_path.unlink()


# ── synthesis ─────────────────────────────────────────────────────────────────

def chapter_to_mp4(kokoro, chapter, mp4_path):
    paragraphs = [p.strip() for p in chapter["text"].split("\n\n") if p.strip()]
    flat = []
    for para in paragraphs:
        segs = make_segments(para)
        for j, seg in enumerate(segs):
            if seg:
                flat.append((seg, j == len(segs) - 1))

    total   = len(flat)
    pieces  = []
    t0      = time.time()

    for i, (seg, is_last) in enumerate(flat):
        try:
            audio, sr = kokoro.create(seg, voice=VOICE, speed=SPEED, lang=LANG)
        except (IndexError, Exception):
            # Segment still too long — split in half and synthesise each part
            mid = len(seg) // 2
            split_at = seg.rfind(" ", 0, mid) or mid
            audio_parts = []
            for part in [seg[:split_at].strip(), seg[split_at:].strip()]:
                if part:
                    a, sr = kokoro.create(part, voice=VOICE, speed=SPEED, lang=LANG)
                    audio_parts.append(a.astype(np.float32))
            audio = np.concatenate(audio_parts) if audio_parts else np.zeros(SAMPLE_RATE, dtype=np.float32)
        pieces.append(audio.astype(np.float32))
        pieces.append(silence(PARA_SILENCE_S if is_last else SENT_SILENCE_S))
        elapsed = time.time() - t0
        rate    = (i + 1) / elapsed if elapsed else 0
        eta     = (total - i - 1) / rate if rate else 0
        print(f"  [{i+1}/{total}] {rate:.2f} seg/s | ETA {eta/60:.1f}m", end="\r", flush=True)

    print()
    wav = mp4_path.with_suffix(".wav")
    sf.write(str(wav), np.concatenate(pieces), SAMPLE_RATE)
    wav_to_mp4(wav, mp4_path)


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
    subprocess.run([ffmpeg, "-y", "-f", "concat", "-safe", "0",
                    "-i", flist, "-c", "copy", str(FINAL_MP4)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    os.unlink(flist)
    print(f"Done -> {FINAL_MP4}  ({FINAL_MP4.stat().st_size/1e6:.0f} MB)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge-only",   action="store_true")
    ap.add_argument("--test-chapter", type=int, default=None)
    args = ap.parse_args()

    CHAPTERS_DIR.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        merge_chapters()
        return

    md       = MD_FILE.read_text(encoding="utf-8")
    chapters = parse_chapters(md)
    print(f"Parsed {len(chapters)} chapters.")

    if args.test_chapter is not None:
        chapters = [chapters[args.test_chapter]]

    remaining = [ch for ch in chapters if not chapter_mp4(ch).exists()]
    print(f"{len(chapters)-len(remaining)} done, {len(remaining)} remaining.")

    if not remaining:
        merge_chapters()
        return

    print("Loading Kokoro on GPU 2 ...")
    kokoro = Kokoro(MODEL_PATH, VOICES_PATH)
    print("Ready.\n")

    for i, ch in enumerate(remaining):
        mp4 = chapter_mp4(ch)
        print(f"[{i+1}/{len(remaining)}] Chapter {ch['num']}: {ch['title']}")
        chapter_to_mp4(kokoro, ch, mp4)
        print(f"  -> {mp4.name}")

    if args.test_chapter is None:
        merge_chapters()


if __name__ == "__main__":
    main()
