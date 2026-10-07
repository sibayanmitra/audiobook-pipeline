#!/usr/bin/env python3
"""Download the Kokoro-82M ONNX weights into ./models.

Uses the kokoro-onnx GitHub release assets (works where huggingface.co is blocked).
The fp32 model is the right one for CPU: it is ~4x faster than the int8 file on AVX-512
Xeons. Pass --int8 to also fetch the small int8 model.
"""
import argparse
import urllib.request
from pathlib import Path

BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
FILES = ["kokoro-v1.0.onnx", "voices-v1.0.bin"]
MODELS = Path(__file__).resolve().parent / "models"


def fetch(name: str):
    dest = MODELS / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"have {name}")
        return
    print(f"downloading {name} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(f"{BASE}/{name}", tmp)
    tmp.replace(dest)
    print(f"  {dest.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8", action="store_true")
    args = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    for f in FILES + (["kokoro-v1.0.int8.onnx"] if args.int8 else []):
        fetch(f)
