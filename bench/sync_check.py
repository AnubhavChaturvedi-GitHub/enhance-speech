"""Checks a video's audio stays in time with the source: compares speech envelopes after export."""
import subprocess, sys, tempfile, json
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import media, dsp
source, merged = Path(sys.argv[1]), Path(sys.argv[2])
folder = tempfile.TemporaryDirectory()
tmp = Path(folder.name)
def starts(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,start_time", "-of", "json", str(p)], capture_output=True, text=True)
    return {s["codec_type"]: float(s.get("start_time", 0)) for s in json.loads(r.stdout)["streams"]}
gains = {"original": 0.0, "enhanced": 0.0, "isolated": 1.0, "music": dsp.strength_gain(100), "background": dsp.strength_gain(100)}
out = tmp / "e.mp4"
media.export_video(merged, source, gains, out)
print("start times: source", starts(source), "merged", starts(merged), "export", starts(out))
def env(p):
    raw = tmp / "x.raw"
    # Time 0 of each file, as a player sees it.
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(p), "-map", "0:a:0", "-ac", "1", "-ar", "48000", "-af", "aresample=async=1:first_pts=0", "-f", "f32le", "-c:a", "pcm_f32le", str(raw)], check=True)
    x = np.fromfile(raw, dtype="float32")
    return x
a, b = env(source), env(out)
n = min(len(a), len(b)); a, b = a[:n], b[:n]
c = fftconvolve(a, b[::-1], mode="same"); mid = len(c) // 2; seg = c[mid - 4800: mid + 4800]; k = int(np.argmax(np.abs(seg))) - 4800
print(f"export audio vs source audio: offset {k} samples = {k/48:.2f} ms; correlation {seg[k+4800]/np.sqrt((a**2).sum()*(b**2).sum()):.3f}")
