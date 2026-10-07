#!/usr/bin/env python3
"""
My Life in Advertising audiobook pipeline: Kokoro-82M (ONNX) on CPU.

Usage:
    python fetch_models.py                       # once: download weights into ./models
    python audiobook_make.py --test-chapter 0 --limit-segments 12   # ~1 minute sample
    python audiobook_make.py                     # whole book, resumable
    python audiobook_make.py --merge-only        # rebuild the .m4b from finished chapters
    python audiobook_make.py --list-voices

On a 4-core CPU the fp32 model runs ~3x realtime in one process, ~4.8x aggregate with
four single-threaded workers, so chapters are synthesised in parallel (--workers).
The int8 model is *slower* than fp32 on CPUs like this one; use kokoro-v1.0.onnx.
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

HERE = Path(__file__).resolve().parent

# ── config ────────────────────────────────────────────────────────────────────
MD_FILE      = HERE / "my_life_in_advertising.md"
OUT_DIR      = HERE / "audiobook"
CHAPTERS_DIR = OUT_DIR / "chapters"
TEST_DIR     = OUT_DIR / "test"
BOOK_TITLE   = "My Life in Advertising"
BOOK_AUTHOR  = "Claude C. Hopkins"
FINAL_M4B    = OUT_DIR / "My_Life_in_Advertising.m4b"

MODEL_PATH  = os.environ.get("KOKORO_MODEL",  str(HERE / "models" / "kokoro-v1.0.onnx"))
VOICES_PATH = os.environ.get("KOKORO_VOICES", str(HERE / "models" / "voices-v1.0.bin"))

VOICE       = "am_echo"
SPEED       = 1.0
LANG        = "en-us"
SAMPLE_RATE = 24000
BITRATE     = "96k"

PARA_SILENCE_S   = 0.55
SENT_SILENCE_S   = 0.15
TITLE_SILENCE_S  = 1.00
SUBHEAD_BEFORE_S = 0.70
SUBHEAD_AFTER_S  = 0.50
MAX_SEG_CHARS    = 250   # kokoro-onnx itself splits anything over 510 phonemes
# ──────────────────────────────────────────────────────────────────────────────

NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen"]


# ── text cleaning ─────────────────────────────────────────────────────────────

_ABBREV_END = re.compile(
    r"(?:\b(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|Co|Inc|Ltd|vs|etc|No|Messrs|Mt|Prof|Gen|Col|Sgt)"
    r"|\b[A-Z])\.$")
_SENT_END = re.compile(r"(?<=[.!?])[\"')”’]*\s+(?=[\"(“‘]*[A-Z0-9])")


def clean(text: str) -> str:
    """Make one paragraph speakable."""
    text = re.sub(r"\bM\. (?=[A-Z])", "Monsieur ", text)          # "M. Pitard"
    text = re.sub(r"^\((\d+)\)\s*", r"\1. ", text)                 # "(3) Brief..." -> "3. Brief..."
    text = re.sub(r"(?:\s*\.){3,}", "...", text)                   # ". . ." -> "..."
    text = text.replace("&", " and ")
    text = re.sub(r"\s*[–—]\s*", ", ", text)         # dashes -> a spoken pause
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+,", ",", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def make_segments(paragraph: str) -> list[str]:
    sentences = []
    for piece in _SENT_END.split(paragraph):
        piece = piece.strip()
        if not piece:
            continue
        if sentences and _ABBREV_END.search(sentences[-1]):      # "Mr." / "J." is not a sentence end
            sentences[-1] += " " + piece
        else:
            sentences.append(piece)
    segments, buf = [], ""
    for sent in sentences:
        if buf and len(buf) + len(sent) + 1 > MAX_SEG_CHARS:
            segments.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        segments.append(buf)
    return segments


# ── chapter parsing ───────────────────────────────────────────────────────────

def spoken_title(heading: str) -> str:
    m = re.match(r"Chapter (\d+): (.*)", heading)
    if not m:
        return heading
    n = int(m.group(1))
    return f"Chapter {NUMBER_WORDS[n] if n < len(NUMBER_WORDS) else n}. {m.group(2)}."


def parse_chapters(md: str) -> list[dict]:
    chapters = []
    for block in re.split(r"^# ", md, flags=re.M)[1:]:
        heading, _, body = block.partition("\n")
        items = []                                # ("title"|"head"|"para", text)
        items.append(("title", spoken_title(heading.strip())))
        for para in re.split(r"\n\s*\n", body):
            para = para.strip()
            if not para:
                continue
            if para.startswith("## "):
                items.append(("head", clean(para[3:])))
            else:
                items.append(("para", clean(para)))
        chapters.append({"num": len(chapters), "title": heading.strip(), "items": items})
    return chapters


def chapter_slug(ch: dict) -> str:
    return f"{ch['num']:02d}_{re.sub(r'[^a-z0-9]+', '_', ch['title'].lower())[:40].strip('_')}"


def chapter_path(ch: dict, directory: Path) -> Path:
    return directory / f"{chapter_slug(ch)}.m4a"


# ── ffmpeg ────────────────────────────────────────────────────────────────────

def find_tool(name: str) -> str:
    path = shutil.which(name)
    if path:
        return path
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
        path = shutil.which(name)
    except ImportError:
        pass
    if not path:
        sys.exit(f"{name} not found. Install ffmpeg (apt install ffmpeg) or pip install static-ffmpeg.")
    return path


def wav_to_m4a(wav: Path, out: Path, title: str, bitrate: str):
    tmp = out.with_name(out.stem + ".part.m4a")           # never leave a truncated chapter behind
    subprocess.run([find_tool("ffmpeg"), "-y", "-i", str(wav), "-c:a", "aac", "-b:a", bitrate,
                    "-metadata", f"title={title}", "-metadata", f"artist={BOOK_AUTHOR}",
                    "-metadata", f"album={BOOK_TITLE}", "-movflags", "+faststart", str(tmp)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    wav.unlink()
    os.replace(tmp, out)


def duration_s(path: Path) -> float:
    out = subprocess.run([find_tool("ffprobe"), "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return float(out.strip())


def merge_chapters(chapters: list[dict]):
    """Concatenate finished chapters into one .m4b with a chapter marker per chapter."""
    done = [(ch, chapter_path(ch, CHAPTERS_DIR)) for ch in chapters]
    done = [(ch, p) for ch, p in done if p.exists()]
    if not done:
        sys.exit("No finished chapters found.")
    if len(done) < len(chapters):
        print(f"WARNING: only {len(done)}/{len(chapters)} chapters finished; merging those.")
    lines, start = [";FFMETADATA1", f"title={BOOK_TITLE}", f"artist={BOOK_AUTHOR}",
                    f"album={BOOK_TITLE}", "genre=Audiobook"], 0
    for ch, p in done:
        end = start + int(duration_s(p) * 1000)
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={start}", f"END={end}",
                  f"title={ch['title']}"]
        start = end
    with tempfile.TemporaryDirectory() as td:
        meta, flist = Path(td) / "meta.txt", Path(td) / "list.txt"
        meta.write_text("\n".join(lines) + "\n", encoding="utf-8")
        flist.write_text("".join(f"file '{p.resolve()}'\n" for _, p in done), encoding="utf-8")
        FINAL_M4B.parent.mkdir(parents=True, exist_ok=True)
        print(f"Merging {len(done)} chapters -> {FINAL_M4B.name}")
        subprocess.run([find_tool("ffmpeg"), "-y", "-f", "concat", "-safe", "0", "-i", str(flist),
                        "-i", str(meta), "-map", "0:a", "-map_metadata", "1", "-c", "copy",
                        "-f", "ipod", str(FINAL_M4B)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    print(f"Done -> {FINAL_M4B}  ({FINAL_M4B.stat().st_size / 1e6:.0f} MB, "
          f"{start / 3.6e6:.2f} h)")


# ── synthesis ─────────────────────────────────────────────────────────────────

_kokoro = None


def load_kokoro(threads: int):
    """One onnxruntime session per process, limited to `threads` CPU threads."""
    import onnxruntime as rt
    from kokoro_onnx import Kokoro
    opts = rt.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    session = rt.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
    return Kokoro.from_session(session, VOICES_PATH)


def _init_worker(threads: int):
    global _kokoro
    _kokoro = load_kokoro(threads)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


def speak(kokoro, text: str, voice: str, speed: float) -> np.ndarray:
    try:
        audio, sr = kokoro.create(text, voice=voice, speed=speed, lang=LANG)
        assert sr == SAMPLE_RATE
        return audio.astype(np.float32)
    except Exception:
        words = text.split()
        if len(words) < 2:
            return silence(0.2)
        mid = len(words) // 2                              # too long for the model: halve and retry
        return np.concatenate([speak(kokoro, " ".join(words[:mid]), voice, speed),
                               silence(0.1),
                               speak(kokoro, " ".join(words[mid:]), voice, speed)])


def synth_chapter(job: tuple) -> tuple:
    """Worker entry point. job = (chapter, out_dir, voice, speed, bitrate, limit_segments)."""
    ch, out_dir, voice, speed, bitrate, limit = job
    kokoro = _kokoro
    plan = []                                              # (text, silence_after, silence_before)
    items = ch["items"]
    for i, (kind, text) in enumerate(items):
        if kind == "title":
            plan.append((text, TITLE_SILENCE_S, 0.0))
        elif kind == "head":
            plan.append((text, SUBHEAD_AFTER_S, SUBHEAD_BEFORE_S))
        else:
            segs = make_segments(text)
            for j, seg in enumerate(segs):
                plan.append((seg, PARA_SILENCE_S if j == len(segs) - 1 else SENT_SILENCE_S, 0.0))
    if limit:
        plan = plan[:limit]

    pieces, t0, next_mark = [], time.time(), 25
    for i, (text, after, before) in enumerate(plan):
        if before:
            pieces.append(silence(before))
        pieces.append(speak(kokoro, text, voice, speed))
        pieces.append(silence(after))
        pct = 100 * (i + 1) // len(plan)
        if pct >= next_mark and pct < 100:
            print(f"  [{ch['num']:02d}] {pct}%  ({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)
            next_mark += 25

    audio = np.concatenate(pieces)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = chapter_path(ch, out_dir)
    wav = out.with_suffix(".wav")
    sf.write(str(wav), audio, SAMPLE_RATE)
    wav_to_m4a(wav, out, ch["title"], bitrate)
    return ch["num"], ch["title"], len(audio) / SAMPLE_RATE, time.time() - t0


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", default=VOICE)
    ap.add_argument("--speed", type=float, default=SPEED)
    ap.add_argument("--bitrate", default=BITRATE)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                    help="parallel chapter workers; CPU threads are split evenly between them")
    ap.add_argument("--test-chapter", type=int, default=None,
                    help="synthesise only this chapter index (0 = Background) into audiobook/test/")
    ap.add_argument("--limit-segments", type=int, default=None,
                    help="stop after N segments (quick sample); implies output in audiobook/test/")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--list-voices", action="store_true")
    args = ap.parse_args()

    if args.list_voices:
        print("\n".join(sorted(np.load(VOICES_PATH).files)))
        return
    if not MD_FILE.exists():
        sys.exit(f"{MD_FILE.name} not found. Run pdf_to_md.py first.")
    chapters = parse_chapters(MD_FILE.read_text(encoding="utf-8"))

    if args.merge_only:
        merge_chapters(chapters)
        return

    for p in (MODEL_PATH, VOICES_PATH):
        if not Path(p).exists():
            sys.exit(f"Missing {p}. Run fetch_models.py first.")
    if args.voice not in np.load(VOICES_PATH).files:
        sys.exit(f"Unknown voice {args.voice!r}. See --list-voices.")

    test_run = args.test_chapter is not None or args.limit_segments is not None
    out_dir = TEST_DIR if test_run else CHAPTERS_DIR
    todo = [chapters[args.test_chapter]] if args.test_chapter is not None else chapters
    if not test_run:
        todo = [ch for ch in todo if not chapter_path(ch, out_dir).exists()]
    print(f"{len(chapters)} chapters, {len(todo)} to do. Voice {args.voice}, "
          f"model {Path(MODEL_PATH).name}.")

    if todo:
        workers = max(1, min(args.workers, len(todo)))
        threads = max(1, (os.cpu_count() or 1) // workers)
        print(f"{workers} worker(s) x {threads} CPU thread(s).\n")
        jobs = [(ch, out_dir, args.voice, args.speed, args.bitrate, args.limit_segments)
                for ch in todo]
        # Longest chapters first so the pool finishes together.
        jobs.sort(key=lambda j: -sum(len(t) for _, t in j[0]["items"]))
        t0, audio_s = time.time(), 0.0
        ctx = mp.get_context("spawn")                      # onnxruntime is not fork-safe
        with ctx.Pool(workers, initializer=_init_worker, initargs=(threads,)) as pool:
            for num, title, dur, took in pool.imap_unordered(synth_chapter, jobs):
                audio_s += dur
                print(f"done [{num:02d}] {title}: {dur / 60:.1f} min audio in {took / 60:.1f} min "
                      f"({dur / took:.1f}x per worker)", flush=True)
        total = time.time() - t0
        print(f"\nSynthesised {audio_s / 3600:.2f} h of audio in {total / 60:.1f} min "
              f"({audio_s / total:.1f}x realtime overall).")

    if test_run:
        print(f"Sample(s) in {out_dir}")
    else:
        merge_chapters(chapters)


if __name__ == "__main__":
    main()
