"""Builds the landing page sample: a public-domain LibriVox reading (Sun Tzu, The Art of War,
read for LibriVox) with room echo and fan noise added, then enhanced by the real pipeline."""
import subprocess, sys, shutil, warnings
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import fftconvolve, butter, sosfilt
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server import pipeline, media, dsp
src = Path(sys.argv[1])
work = ROOT / "bench" / "work" / "landing"
shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
clean_wav = work / "clean.wav"
subprocess.run(["ffmpeg","-y","-v","error","-ss",sys.argv[2],"-t",sys.argv[3],"-i",str(src),"-ac","1","-ar","48000","-c:a","pcm_f32le",str(clean_wav)],check=True)
x, sr = sf.read(clean_wav, dtype="float32")
rng = np.random.default_rng(7)
n = int(sr * 0.55)
rir = rng.standard_normal(n) * np.exp(-np.arange(n) / (0.09 * sr))
rir[0] = 3.2
rir = rir / np.sqrt((rir**2).sum())
wet = fftconvolve(x, rir)[: len(x)]
noise = rng.standard_normal(len(x))
fan = sosfilt(butter(2, [80, 900], btype="band", fs=sr, output="sos"), noise)
hiss = sosfilt(butter(2, 3000, btype="high", fs=sr, output="sos"), noise[::-1])
speech_rms = np.sqrt((wet**2).mean())
y = wet + fan / np.sqrt((fan**2).mean()) * speech_rms * 10 ** (-20 / 20) + hiss / np.sqrt((hiss**2).mean()) * speech_rms * 10 ** (-32 / 20)
y = (y / max(1e-6, np.abs(y).max()) * 0.5).astype(np.float32)
dirty = work / "dirty.wav"
sf.write(dirty, y, sr, subtype="FLOAT")
out = work / "merged.mp4"
meta = pipeline.process(dirty, work / "w", out, "v2", lambda s, f: None)
print(meta)
dest = ROOT / "web" / "assets" / "sample"
dest.mkdir(parents=True, exist_ok=True)
gains = {"original": 0.0, "enhanced": 0.9, "isolated": 0.1, "music": dsp.strength_gain(10), "background": dsp.strength_gain(10)}
media.export_audio(out, gains, work / "enhanced.wav")
for name, wav, layout in [("original", dirty, ["-af", "pan=stereo|c0=c0|c1=c0"]), ("enhanced", work / "enhanced.wav", [])]:
    subprocess.run(["ffmpeg","-y","-v","error","-i",str(wav),*layout,"-c:a","libmp3lame","-b:a","160k",str(dest / f"{name}.mp3")],check=True)
print("done")
