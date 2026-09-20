import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel

MODEL = "./models/Qwen3-TTS-12Hz-1.7B-CustomVoice"
model = Qwen3TTSModel.from_pretrained(
    MODEL, device_map="cuda:0", dtype=torch.bfloat16,
)
wavs, sr = model.generate_custom_voice(
    text="Hello! This is Qwen3 text to speech, running locally on the GPU.",
    language="English",
    speaker="Ryan",
)
sf.write("sample_output.wav", wavs[0], sr)
print(f"OK -> sample_output.wav  | sample_rate={sr}  samples={len(wavs[0])}  dur={len(wavs[0])/sr:.2f}s")
