#!/usr/bin/env python3
"""Package a finished audiobook as several small .m4b parts (whole chapters per part).

Usage:
    python split_for_sharing.py ogilvy_confessions [--bitrate 64k] [--max-mib 28]

Reads the per-chapter .m4a files in <book>/audiobook/chapters, groups consecutive chapters
until a part would exceed --max-mib, and re-encodes each group (mono) into
<book>/audiobook/parts/<Title>_part_N_of_M.m4b with a chapter marker per chapter.
Useful when the whole book is too big to upload or attach in one piece.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("book")
    ap.add_argument("--bitrate", default="64k")
    ap.add_argument("--max-mib", type=float, default=28.0)
    args = ap.parse_args()

    book = ROOT / args.book
    sys.path.insert(0, str(book))
    os.chdir(book)
    import audiobook_make as am

    kbps = int(args.bitrate.rstrip("k"))
    budget = args.max_mib * 1024 * 1024 * 0.95 / (kbps * 125)          # seconds per part, 5% headroom
    chapters = [(ch, am.chapter_path(ch, am.CHAPTERS_DIR)) for ch in
                am.parse_chapters(am.MD_FILE.read_text(encoding="utf-8"))]
    chapters = [(ch, p, am.duration_s(p)) for ch, p in chapters if p.exists()]
    if not chapters:
        sys.exit("No finished chapters found.")

    groups, cur, cur_s = [], [], 0.0
    for item in chapters:
        if cur and cur_s + item[2] > budget:
            groups.append(cur)
            cur, cur_s = [], 0.0
        cur.append(item)
        cur_s += item[2]
    groups.append(cur)

    out_dir = am.OUT_DIR / "parts"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("*.m4b"):
        old.unlink()
    stem = am.FINAL_M4B.stem
    for n, group in enumerate(groups, 1):
        meta, start = [";FFMETADATA1", f"title={am.BOOK_TITLE} (part {n} of {len(groups)})",
                       f"artist={am.BOOK_AUTHOR}", f"album={am.BOOK_TITLE}", "genre=Audiobook",
                       f"track={n}/{len(groups)}"], 0
        for ch, _, dur in group:
            end = start + int(dur * 1000)
            meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={start}", f"END={end}", f"title={ch['title']}"]
            start = end
        out = out_dir / f"{stem}_part_{n}_of_{len(groups)}.m4b"
        with tempfile.TemporaryDirectory() as td:
            mpath, lpath = Path(td) / "meta.txt", Path(td) / "list.txt"
            mpath.write_text("\n".join(meta) + "\n", encoding="utf-8")
            lpath.write_text("".join(f"file '{p.resolve()}'\n" for _, p, _ in group), encoding="utf-8")
            subprocess.run([am.find_tool("ffmpeg"), "-y", "-f", "concat", "-safe", "0", "-i", str(lpath),
                            "-i", str(mpath), "-map", "0:a", "-map_metadata", "1", "-c:a", "aac",
                            "-b:a", args.bitrate, "-ac", "1", "-f", "ipod", str(out)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        mib = out.stat().st_size / 1024 / 1024
        print(f"{out.name}: {len(group)} chapter(s), {sum(d for *_, d in group) / 60:.1f} min, {mib:.1f} MiB")
        if mib > 29.5:
            sys.exit(f"{out.name} is {mib:.1f} MiB - lower --max-mib or --bitrate")


if __name__ == "__main__":
    main()
