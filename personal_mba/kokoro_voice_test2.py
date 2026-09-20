import numpy as np
import soundfile as sf
import subprocess, shutil, os
import static_ffmpeg
from kokoro_onnx import Kokoro

MODEL  = "/home/sibayan_mitra_2024/audio/models/kokoro-onnx/kokoro-v1.0.fp16.onnx"
VOICES = "/home/sibayan_mitra_2024/audio/models/kokoro-onnx/voices-v1.0.bin"

kokoro = Kokoro(MODEL, VOICES)

remaining = ["am_liam", "am_michael", "am_onyx", "am_puck", "am_santa"]

text = ("Every business is a collection of processes — repeatable, learnable systems. "
        "Understanding them is the foundation of every lasting success.")

static_ffmpeg.add_paths()
ffmpeg = shutil.which("ffmpeg")

for voice in remaining:
    samples, sr = kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
    wav = f"/tmp/{voice}.wav"
    mp4 = f"voice_test_{voice}.mp4"
    sf.write(wav, samples, sr)
    subprocess.run([ffmpeg, "-y", "-i", wav, "-c:a", "aac", "-b:a", "128k", mp4],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.unlink(wav)
    print(f"-> {mp4}")

print("Done.")
