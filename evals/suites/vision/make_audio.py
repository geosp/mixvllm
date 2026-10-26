"""Generate a 3 s, 16 kHz mono WAV: three 0.3 s beeps at rising pitch (440, 660, 880 Hz), then silence."""

import math
import struct
import sys
import wave

RATE = 16000
samples = []
for freq in (440, 660, 880):
    samples += [int(12000 * math.sin(2 * math.pi * freq * i / RATE)) for i in range(int(0.3 * RATE))]
    samples += [0] * int(0.4 * RATE)
samples += [0] * (3 * RATE - len(samples))
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(RATE)
    w.writeframes(struct.pack(f"<{len(samples)}h", *samples))
print("samples:", len(samples))
