# Audiobooks on a plain Intel CPU (no GPU)

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) is a small (82M-parameter) open-weights
text-to-speech model, and its ONNX build runs faster than realtime on an ordinary CPU. This
branch packages the CPU path of this repo so you can turn your own PDFs into audiobooks on a
laptop, desktop or cheap cloud VM.

The worked example is `ogilvy_confessions/` (a two-column PDF -> chapters -> one `.m4b` with
chapter markers).

## Quick start

```bash
cd ogilvy_confessions
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt            # pymupdf, pyspellchecker, kokoro-onnx, soundfile, numpy
sudo apt install ffmpeg                    # or: pip install static-ffmpeg
python fetch_models.py                     # ~350 MB, from GitHub releases (not huggingface.co)

mkdir -p raw && cp /path/to/book.pdf raw/
python pdf_to_md.py                        # -> confessions.md + chapters.json
python audiobook_make.py --test-chapter 0 --limit-segments 10   # ~1 min sample in audiobook/test/
python audiobook_make.py                   # whole book, resumable -> audiobook/*.m4b
```

`python audiobook_make.py --list-voices` prints the 54 voices. `--voice bm_george` is a British
male voice; `am_echo` is the default.

## What was measured

One machine, so treat these as a guide, not a guarantee: 4 vCPU Intel Xeon @ 2.1 GHz (AVX-512 +
VNNI), 15 GB RAM, no GPU. Speed is audio seconds produced per wall-clock second, on a ~19 s
passage of book text, `kokoro-v1.0.onnx` unless noted.

| setup | speed |
|---|---|
| 1 process, 4 threads | 3.0-3.1x realtime |
| 1 process, 2 threads | 2.2x |
| 1 process, 1 thread | 1.27x |
| 2 processes x 2 threads (concurrent) | ~4.3x combined |
| **4 processes x 1 thread (concurrent)** | **~4.8x combined** |
| `kokoro-v1.0.int8.onnx`, 4 threads | 0.8x (slower!) |

Take-aways:

* **Use the fp32 model on CPU.** The int8 file is smaller but was ~4x *slower* here.
* **Several single-threaded workers beat one multi-threaded process** (about 55% more
  throughput), because small TTS batches don't scale across threads. `audiobook_make.py` does this
  by default: `--workers` (default `min(4, cores)`), with CPU threads split evenly.
* Each worker uses ~1 GB of RAM.
* Rough planning figure, extrapolated and **not tested beyond 4 cores**: about 1.2x realtime
  per core, so audio length / (1.2 x cores) is the wall-clock time.
* Only an AVX-512 Intel CPU was tested. It should run on any CPU onnxruntime supports
  (AVX2 x86, Apple Silicon, ARM), but speeds there are unmeasured.

## Why the pipeline looks the way it does

* **No huggingface.co needed.** `fetch_models.py` downloads the weights from the kokoro-onnx
  GitHub releases, which works on networks that block Hugging Face.
* **Chapter-level resume.** Each chapter is written to a temporary file and renamed when
  complete, so Ctrl-C or a crash never leaves a truncated chapter; re-running skips finished ones.
* **Text repair before speech.** `pdf_to_md.py` removes running headers, page numbers and
  footnotes, reads two columns in order, rejoins paragraphs that wrap across columns and pages,
  and decides line-end hyphens with a dictionary ("dis-inherited" -> "disinherited",
  "self-advertisement" stays hyphenated).
* **Chapters come from the PDF's own table of contents** (`doc.get_toc()`), so a PDF without
  bookmarks needs a hand-written `chapters.json`.

## Using it on a different book

`pdf_to_md.py` has two kinds of constants you will want to look at:

* the **page geometry** at the top (header/page-number/footnote bands, column split, font sizes)
  is specific to this PDF's layout. Dump `page.get_text("dict")` for a page of your book and
  adjust them. A single-column book is simpler: set `COLUMN_SPLIT_X` beyond the page width.
* **`FIXUPS`** is a list of regex repairs for OCR errors and two charts in *this* book. Replace
  it with fixes for yours (or empty it). A spell-check pass over the generated markdown is a
  quick way to find your own OCR errors.

Then check the generated `.md` by eye before spending an hour of CPU on it.

## Please respect copyright

This repo contains only scripts. Convert books you own or have the right to convert, and don't
publish the generated text or audio of copyrighted works. Kokoro's own license applies to the
model and voices.
