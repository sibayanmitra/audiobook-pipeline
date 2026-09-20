# Audiobook Pipeline

Scripts for turning PDF books into audiobooks: PDF → markdown → chunked text → TTS.

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

Each per-book directory is a copy of the pipeline adapted to that book's structure
(`titan/chapters.json` is an example chapter map). **No book text or audio is included in
this repository** — only the conversion scripts. Point them at your own source PDFs.
