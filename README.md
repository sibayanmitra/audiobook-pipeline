# Audiobook Pipeline

Scripts for turning PDF books into audiobooks (CPU-only quick start for Intel machines: see [INTEL_CPU.md](INTEL_CPU.md)): PDF → markdown → chunked text → TTS.

## Stages

| script | role |
|---|---|
| `pdf_to_md.py` | PDF text extraction and markdown conversion |
| `chunk_md.py` | split markdown into TTS-sized chunks at sentence boundaries |
| `audiobook_make.py` | drive the TTS engine over chunks and stitch the output |
| `audiobook_make_qwen.py` | Qwen-TTS variant |
| `smoke_test.py` | end-to-end sanity check |

`personal_mba/kokoro_setup.py`, `kokoro_voice_test*.py`, and `benchmark_gpu.py` cover
Kokoro voice selection and GPU throughput benchmarking.

`ogilvy_confessions/` is the CPU-only variant (no GPU needed): `pdf_to_md.py` handles a
two-column PDF, `fetch_models.py` downloads the Kokoro-82M ONNX weights from GitHub releases
(useful where huggingface.co is blocked), and `audiobook_make.py` runs Kokoro on CPU with
parallel chapter workers (~4.8x realtime on 4 cores), resumes per chapter and merges the
result into a single `.m4b` with chapter markers. The fp32 model is faster than int8 on CPU.

Each per-book directory is a copy of the pipeline adapted to that book's structure
(`titan/chapters.json` is an example chapter map). **No book text or audio is included in
this repository** — only the conversion scripts. Point them at your own source PDFs.
