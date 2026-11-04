"""Looks for clicks or level jumps at the pipeline's block seams, compared with random positions."""
import subprocess, sys
from pathlib import Path
import numpy as np
merged, *blocks = sys.argv[1:]
blocks = [int(b) for b in blocks]
raw = Path(merged).with_suffix(".enh.raw")
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", merged, "-map", "0:a:0", "-af", "pan=mono|c0=c2", "-f", "f32le", "-c:a", "pcm_f32le", str(raw)], check=True)
x = np.fromfile(raw, dtype="float32"); raw.unlink()
sr = 48000
def click(t):
    i = int(t * sr); w = x[i - 240: i + 240]
    if len(w) < 480: return None
    d = np.abs(np.diff(w)).max()
    rms = np.sqrt((x[i - 2400: i + 2400] ** 2).mean()) + 1e-6
    return d / rms
speech_level = 20 * np.log10(np.percentile(np.sqrt(np.convolve(x[::48] ** 2, np.ones(20) / 20, mode="same")), 95) + 1e-6)
def level_jump(t):
    """Level step across t, or None where either side is in a gate's closed gap, because a word onset there is not a fault."""
    i = int(t * sr)
    a = 20 * np.log10(np.sqrt((x[i - 1440: i - 240] ** 2).mean()) + 1e-6)
    b = 20 * np.log10(np.sqrt((x[i + 240: i + 1440] ** 2).mean()) + 1e-6)
    return abs(a - b) if min(a, b) > speech_level - 38 else None
rng = np.random.default_rng(1)
dur = len(x) / sr
if dur < 2:
    sys.exit(f"{dur:.1f} s is too short to have block seams")
rand = rng.uniform(1, dur - 1, 800)
base = np.array([v for v in (click(t) for t in rand) if v is not None])
jump = np.array([v for v in (level_jump(t) for t in rand) if v is not None])
print(f"random positions: click p50 {np.percentile(base,50):.2f} p99 {np.percentile(base,99):.2f} max {base.max():.2f}; level jump p50 {np.percentile(jump,50):.1f} dB p99 {np.percentile(jump,99):.1f} dB")
for b in blocks:
    seams = np.arange(b, dur - 1, b)
    if not len(seams):
        print(f"seams every {b:>3}s: none in a {dur:.0f} s file")
        continue
    c = np.array([click(t) for t in seams]); j = np.array([v for v in (level_jump(t) for t in seams) if v is not None])
    steps = f"level jump median {np.median(j):.1f} dB max {j.max():.1f} dB on {len(j)} not in a gap" if len(j) else "every seam in a gap"
    print(f"seams every {b:>3}s (n={len(seams)}): click median {np.median(c):.2f} max {c.max():.2f}; above random p99: {(c > np.percentile(base,99)).sum()}; {steps}")
