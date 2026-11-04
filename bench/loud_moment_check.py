"""Checks that one loud moment leaves the rest of a recording alone: runs the real chain on a clip with and without a word added close to the mic.

Usage: loud_moment_check.py <speech recording> <near talker recording> [dB over the speech, default 30]
Prints the speech's loudness away from the moment, and the tone it moved to, against the clean run. Stems stay in bench/work/loud.
"""
import shutil, subprocess, sys
from pathlib import Path
import numpy as np, soundfile as sf
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server import gate, level, pipeline
SR, AT, SECONDS, CLIP = 48000, 14.0, 1.0, 30
BANDS = [(50, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 12000), (12000, 16000)]


def decode(path, start=0.0):
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", str(start), "-t", str(CLIP), "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32).copy()


def p95(x):
    return float(np.percentile(gate._levels(x, 960, 480), 95))


def band_db(x):
    size, hop = 1920, 480
    frames = np.stack([x[i : i + size] * np.hanning(size) for i in range(0, len(x) - size, hop)])
    s = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    f = np.fft.rfftfreq(size, 1 / SR)
    t = 10 * np.log10(s.sum(axis=1) + 1e-20)
    on = t > np.percentile(t, 95) - 30
    return 10 * np.log10(np.stack([s[:, (f >= a) & (f < b)][on].sum() for a, b in BANDS]) / s[on].sum() + 1e-20)


speech, near = decode(sys.argv[1]), decode(sys.argv[2])
over = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
power = np.convolve(near.astype(np.float64) ** 2, np.ones(4800) / 4800, "same")
word = near[max(0, int(np.argmax(power)) - SR // 2) :][: int(SECONDS * SR)]
loud = speech.copy()
at = int(AT * SR)
loud[at : at + len(word)] += word * 10 ** ((p95(speech) + over - p95(word)) / 20)
loud *= min(1.0, 0.9 / np.abs(loud).max())

base = ROOT / "bench" / "work" / "loud"
shutil.rmtree(base, ignore_errors=True)
base.mkdir(parents=True)
out = {}
for tag, x in (("clean", speech), ("loud", loud)):
    (base / tag).mkdir()
    sf.write(base / tag / "in.wav", x, SR, subtype="FLOAT")
    pipeline.process(base / tag / "in.wav", base / tag / "work", base / tag / "merged.mp4", "v2", lambda s, f: None)
    out[tag] = np.fromfile(base / tag / "work" / "enhanced.raw", np.float32)
keep = np.ones(len(speech), bool)
keep[int((AT - 1.5) * SR) : int((AT + SECONDS + 1.5) * SR)] = False
print(f"speech away from the moment: {level.integrated_lufs(out['loud'][keep]):.2f} LUFS, clean run {level.integrated_lufs(out['clean'][keep]):.2f}")
print("tone against the clean run, dB: " + " ".join(f"{a}:{d:+.1f}" for (a, _), d in zip(BANDS, band_db(out["loud"][keep]) - band_db(out["clean"][keep]))))
moment = out["loud"][at : at + len(word)]
print(f"the moment against the speech's level: input +{over:.0f} dB, output {p95(moment) - p95(out['loud'][keep]):+.1f} dB")
