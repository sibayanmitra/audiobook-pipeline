#!/usr/bin/env python3
"""Run several books through the CPU pipeline one after another, with sanity checks.

Usage:
    python queue_books.py ogilvy_confessions another_book ...

Each book directory must contain raw/*.pdf and the CPU pipeline scripts
(pdf_to_md.py, audiobook_make.py, check_audiobook.py). Per book, in order:

    1. pdf_to_md.py              extract text
    2. check_audiobook.py --text   FAIL here -> skip synthesis (don't burn hours of CPU on bad text)
    3. audiobook_make.py         synthesise + merge into one .m4b (resumable)
    4. check_audiobook.py --audio  verify durations, silence, clipping, chapter markers

A failing book is recorded and the queue moves on to the next one. If another
audiobook_make.py is already running, the queue waits for it so books never share the CPU.
Progress is appended to queue.log in the repo root; the last lines are a summary table.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "queue.log"


def log(msg: str):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def run(book: Path, *cmd: str) -> bool:
    log(f"  $ {' '.join(cmd)}")
    with LOG.open("a") as f:
        rc = subprocess.run([sys.executable, *cmd], cwd=book, stdout=f, stderr=subprocess.STDOUT).returncode
    return rc == 0


def wait_for_cpu():
    announced = False
    while subprocess.run(["pgrep", "-f", "audiobook_make.py"], capture_output=True).returncode == 0:
        if not announced:
            log("waiting for the running audiobook_make.py to finish ...")
            announced = True
        time.sleep(30)


def main():
    books = [ROOT / b for b in sys.argv[1:]]
    if not books:
        sys.exit(__doc__)
    summary = []
    for book in books:
        t0 = time.time()
        log(f"=== {book.name} ===")
        if not list((book / "raw").glob("*.pdf")):
            summary.append((book.name, "SKIPPED: no raw/*.pdf"))
            log(summary[-1][1])
            continue
        wait_for_cpu()
        if not run(book, "pdf_to_md.py"):
            summary.append((book.name, "FAILED: pdf_to_md.py"))
        elif not run(book, "check_audiobook.py", "--text"):
            summary.append((book.name, "NEEDS REVIEW: text checks failed (no audio made)"))
        elif not run(book, "audiobook_make.py"):
            summary.append((book.name, "FAILED: audiobook_make.py (rerun resumes)"))
        elif not run(book, "check_audiobook.py", "--audio"):
            summary.append((book.name, "NEEDS REVIEW: audio checks failed"))
        else:
            summary.append((book.name, f"OK in {(time.time() - t0) / 60:.0f} min"))
        log(f"{book.name}: {summary[-1][1]}")
    log("=== summary ===")
    for name, status in summary:
        log(f"{name:<28} {status}")


if __name__ == "__main__":
    main()
