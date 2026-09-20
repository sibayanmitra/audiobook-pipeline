from kokoro_onnx import Kokoro

k = Kokoro.from_pretrained("hexgrad/Kokoro-82M", local_dir="/home/sibayan_mitra_2024/audio/models/kokoro-82M")
voices = k.get_voices()
print(f"Model loaded. {len(voices)} voices available:")
for v in sorted(voices):
    print(" ", v)
