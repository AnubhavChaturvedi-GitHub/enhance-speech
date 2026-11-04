"""Scores the final pipeline on each recording at the slider positions that matter, mixed at the player's gains."""
import sys, shutil, subprocess, warnings, difflib, re
from pathlib import Path
import numpy as np, soundfile as sf, librosa
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server import pipeline, dsp
from speechmos import dnsmos
from faster_whisper import WhisperModel
wm = WhisperModel("small", device="cpu", compute_type="int8")
def text(x):
    y = librosa.resample(x, orig_sr=48000, target_sr=16000)
    s, _ = wm.transcribe(y.astype(np.float32), language="en", beam_size=1, condition_on_previous_text=False)
    return " ".join(i.text.strip() for i in s)
words = lambda t: re.sub(r"[^\w\s]", "", t.lower()).split()
def ds(x):
    y = librosa.resample(x, orig_sr=48000, target_sr=16000); y = y / max(1e-6, np.abs(y).max())
    r = dnsmos.run(y.astype(np.float32), 16000); return r["ovrl_mos"], r["sig_mos"], r["bak_mos"], r["p808_mos"]
def mix(x, e, b=10, m=10):
    k = e / 100
    return k * x[:, 2] + (1 - k) * x[:, 3] + dsp.strength_gain(m) * x[:, 4:6].mean(axis=1) + dsp.strength_gain(b) * x[:, 6:8].mean(axis=1)
cases = [a.split("=", 1) for a in sys.argv[1:]]
for tag, path in cases:
    ref = None
    if "|" in path:
        path, ref = path.split("|")
    work = ROOT / "bench" / "work" / f"final_{tag}"
    shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
    out = work / "merged.mp4"
    pipeline.process(Path(path), work / "w", out, "v2", lambda s, f: None)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(out), "-map", "0:a:0", "-c:a", "pcm_f32le", str(work / "m.wav")], check=True)
    x = sf.read(work / "m.wav", dtype="float32", always_2d=True)[0]
    orig = x[:, :2].mean(axis=1)
    if ref:
        rr = sf.read(ref, dtype="float32", always_2d=True)[0].mean(axis=1)
        base = text(rr)
    else:
        base = text(orig)
    print(f"\n{tag}   {'OVRL':>6}{'SIG':>6}{'BAK':>6}{'P808':>6}{'words':>7}")
    for name, y in [("original", orig), ("speech 100%, rest off", mix(x, 100, 0, 0)), ("speech 90 / 10 / 10", mix(x, 90)), ("default 50 / 10 / 10", mix(x, 50)), ("speech 0 / 10 / 10", mix(x, 0))]:
        s = ds(y)
        print(f"{name:<24}{s[0]:6.2f}{s[1]:6.2f}{s[2]:6.2f}{s[3]:6.2f}{difflib.SequenceMatcher(None, words(base), words(text(y))).ratio():7.3f}", flush=True)
    shutil.rmtree(work / "w", ignore_errors=True)
