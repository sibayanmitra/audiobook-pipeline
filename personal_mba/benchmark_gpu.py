import time, os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider"

from kokoro_onnx import Kokoro

MODEL  = "/home/sibayan_mitra_2024/audio/models/kokoro-onnx/kokoro-v1.0.fp16.onnx"
VOICES = "/home/sibayan_mitra_2024/audio/models/kokoro-onnx/voices-v1.0.bin"

print("Loading model on CUDA:2 ...")
kokoro = Kokoro(MODEL, VOICES)

text = ("Mental Models are the foundation of clear thinking. A mental model is a representation "
        "of how something works. We cannot keep all of the details of the world in our brains, "
        "so we use models to simplify the complex into understandable and organizable chunks. "
        "The quality of your thinking is largely determined by the mental models in your head "
        "and your ability to use them. The best mental models are the ideas with the most "
        "utility — the ones that are applicable across multiple domains, that are durable "
        "over time, and that are most closely aligned with reality as it actually exists.")

# warmup
kokoro.create(text[:100], voice="am_echo", speed=1.0, lang="en-us")

t0 = time.time()
samples, sr = kokoro.create(text, voice="am_echo", speed=1.0, lang="en-us")
elapsed = time.time() - t0
audio_dur = len(samples) / sr

print(f"Audio output : {audio_dur:.1f}s")
print(f"Compute time : {elapsed:.2f}s")
print(f"Speed        : {audio_dur/elapsed:.1f}x real-time")

book_chars = 973504
scale = book_chars / len(text)
est_hrs = (elapsed * scale) / 3600
print(f"\nFull book estimate: ~{est_hrs:.1f} hours  ({est_hrs*60:.0f} mins)")
