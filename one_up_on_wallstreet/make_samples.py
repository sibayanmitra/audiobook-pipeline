#!/usr/bin/env python3
"""Generate short voice samples so we can pick a narrator."""
from pathlib import Path
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE.parent / "models" / "Qwen3-TTS-12Hz-1.7B-CustomVoice"
OUT = HERE / "samples"
OUT.mkdir(exist_ok=True)

TEXT = ("Average investors can become experts in their own field and can pick "
        "winning stocks as effectively as Wall Street professionals, by doing "
        "just a little research.")

# English natives first, then a few others for variety
SPEAKERS = ["Ryan", "Aiden", "Dylan", "Eric", "Vivian", "Serena"]

model = Qwen3TTSModel.from_pretrained(
    str(MODEL_PATH), device_map="cuda:0", dtype=torch.bfloat16,
)
print("Model ready.", flush=True)

wavs, sr = model.generate_custom_voice(
    text=[TEXT] * len(SPEAKERS),
    language=["English"] * len(SPEAKERS),
    speaker=SPEAKERS,
)
for spk, w in zip(SPEAKERS, wavs):
    p = OUT / f"sample_{spk}.wav"
    sf.write(str(p), w, sr)
    print(f"wrote {p}", flush=True)
print("DONE", flush=True)
