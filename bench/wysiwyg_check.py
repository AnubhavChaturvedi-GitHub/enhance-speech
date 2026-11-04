"""Confirms a downloaded mix equals the player's gain formula, with the mono speech stems 3 dB down in v2, as in the reference downloads."""
import subprocess, sys, tempfile
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import dsp, level, media
merged = Path(sys.argv[1])
rms = lambda a: 20 * np.log10(np.sqrt((a ** 2).mean()) + 1e-12)
with tempfile.TemporaryDirectory() as folder:
    tmp = Path(folder)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(merged), "-map", "0:a:0", "-c:a", "pcm_f32le", str(tmp / "m.wav")], check=True)
    m = sf.read(tmp / "m.wav", dtype="float32", always_2d=True)[0]
    for mode, speech, bg, mus in [("v2", 75, 30, 20), ("v2", 100, 0, 0), ("v2", 0, 100, 100), ("v2", 50, 10, 10), ("v1", 75, 0, 0), ("v1", 100, 0, 0)]:
        k = speech / 100
        media.export_audio(merged, dsp.download_gains(speech, False, bg, False, mus, False, mode == "v1"), tmp / "e.wav")
        e = sf.read(tmp / "e.wav", dtype="float32", always_2d=True)[0]
        if mode == "v1":
            left, right = (1 - k) * m[:, 0] + k * m[:, 2], (1 - k) * m[:, 1] + k * m[:, 2]
        else:
            voice = (k * m[:, 2] + (1 - k) * m[:, 3]) * level.EXPORT_SPEECH_GAIN
            left = voice + dsp.strength_gain(mus) * m[:, 4] + dsp.strength_gain(bg) * m[:, 6]
            right = voice + dsp.strength_gain(mus) * m[:, 5] + dsp.strength_gain(bg) * m[:, 7]
        n = min(len(left), len(e))
        diff = np.stack([left[:n], right[:n]], 1) - e[:n]
        print(f"{mode} speech {speech} bg {bg} music {mus}: length {len(e)} vs {len(left)}; difference {rms(diff):.1f} dBFS against a mix at {rms(e):.1f} dBFS; peak {np.abs(e).max():.3f} (formula peak {max(np.abs(left).max(), np.abs(right).max()):.3f})")
