"""Compares our merged stems with Adobe's, channel by channel, from each tool's 8-channel file."""
import sys, subprocess, warnings
from pathlib import Path
import numpy as np, soundfile as sf, librosa
from scipy.signal import fftconvolve, butter, sosfiltfilt
warnings.filterwarnings("ignore")
ours_path, adobe_path = sys.argv[1], sys.argv[2]

def load8(p):
    tmp = Path(p).with_suffix(".8ch.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p, "-map", "0:a:0", "-c:a", "pcm_f32le", str(tmp)], check=True)
    x, _ = sf.read(tmp, dtype="float32", always_2d=True)
    tmp.unlink()
    return x

o, a = load8(ours_path), load8(adobe_path)
n = min(len(o), len(a)); o, a = o[:n], a[:n]
sr = 48000
def sdr(e, r):
    k = (e*r).sum()/max(1e-9, (r*r).sum())
    return 10*np.log10((k*k*(r*r).sum())/max(1e-9, ((e-k*r)**2).sum()))
def mel(x):
    return 10*np.log10(librosa.feature.melspectrogram(y=x, sr=sr, n_fft=2048, hop_length=480, n_mels=128, fmax=24000)+1e-10)
def lsd(e, r):
    g = np.sqrt((r**2).mean()/max(1e-12, (e**2).mean()))
    me, mr = mel(e*g), mel(r)
    en = mr.max(axis=0); act = en > en.max()-45
    return float(np.mean(np.sqrt(np.mean((me[:,act]-mr[:,act])**2, axis=0))))
def dnsmos(x):
    from speechmos import dnsmos as d
    y = librosa.resample(x, orig_sr=sr, target_sr=16000); y = y/max(1e-6, np.abs(y).max())
    r = d.run(y.astype(np.float32), 16000); return r["ovrl_mos"], r["sig_mos"], r["bak_mos"], r["p808_mos"]
def loud(x):
    import pyloudnorm as pl
    return pl.Meter(sr).integrated_loudness(x.astype(np.float64))
def band(c, lo, hi): return sosfiltfilt(butter(4, [lo, hi], btype="band", fs=sr, output="sos"), c)
def ncc(a_, b_, m=3000):
    c = fftconvolve(a_, b_[::-1], mode="same"); mid = len(c)//2; seg = c[mid-m:mid+m]; k = np.argmax(np.abs(seg))
    return float(seg[k]/np.sqrt((a_**2).sum()*(b_**2).sum()))

print("STEM         SI-SDR vs Adobe   LSD vs Adobe   ours LUFS  Adobe LUFS")
for name, oc, ac in [("enhanced", o[:,2], a[:,2]), ("isolated", o[:,3], a[:,3]), ("music", o[:,4:6].mean(axis=1), a[:,4:6].mean(axis=1)), ("background", o[:,6:8].mean(axis=1), a[:,6:8].mean(axis=1))]:
    print(f"{name:<12}{sdr(oc,ac):>12.2f}{lsd(oc,ac):>15.2f}{loud(oc):>13.1f}{loud(ac):>11.1f}")
def mix(x, e, b=10, m=10):
    from_gain = lambda p: 0.0 if p == 0 else (1.0 if p == 100 else 10**(-36*(1-p/100)**2/20))
    k = e/100
    return k*x[:,2] + (1-k)*x[:,3] + from_gain(m)*x[:,4:6].mean(axis=1) + from_gain(b)*x[:,6:8].mean(axis=1)
print("\nMIX                  LSD vs Adobe's same mix   OVRL  SIG  BAK  P808 | Adobe OVRL SIG BAK P808")
for e, b, m in [(100,0,0), (90,10,10), (50,10,10), (0,10,10), (50,100,100), (0,100,100)]:
    mo, ma = mix(o,e,b,m), mix(a,e,b,m)
    so, sa = dnsmos(mo), dnsmos(ma)
    print(f"speech {e:>3}% bg {b:>3}% mus {m:>3}%   {lsd(mo,ma):>8.2f}          {so[0]:.2f} {so[1]:.2f} {so[2]:.2f} {so[3]:.2f} | {sa[0]:.2f} {sa[1]:.2f} {sa[2]:.2f} {sa[3]:.2f}")
print("\nCOHERENCE enhanced vs isolated:   full | <1k | 1-4k | 4-8k")
for nm, x in [("ours", o), ("Adobe", a)]:
    e, i = x[:,2], x[:,3]
    print(f"  {nm:<8}", " | ".join(f"{v:.3f}" for v in [ncc(e,i), ncc(band(e,100,1000),band(i,100,1000)), ncc(band(e,1000,4000),band(i,1000,4000)), ncc(band(e,4000,8000),band(i,4000,8000))]))
print("\nFLOOR (RMS percentiles p5/p20/p50/p95 dBFS)")
def pct(c):
    db = 20*np.log10(librosa.feature.rms(y=c, frame_length=2400, hop_length=480)[0]+1e-9)
    return " ".join(f"{v:7.1f}" for v in np.percentile(db,[5,20,50,95]))
for nm, oc, ac in [("enhanced", o[:,2], a[:,2]), ("isolated", o[:,3], a[:,3])]:
    print(f"  {nm:<10} ours {pct(oc)}   Adobe {pct(ac)}")
print("\nBANDS dB:    30-100 100-300 300-1k  1-3k  3-6k 6-10k 10-16k 16-24k")
def ltas(c):
    S = np.abs(librosa.stft(c, n_fft=4096, hop_length=1024))**2; f = librosa.fft_frequencies(sr=sr, n_fft=4096); v = S.mean(axis=1)
    return " ".join(f"{10*np.log10(v[(f>=lo)&(f<hi)].mean()+1e-14):6.1f}" for lo,hi in [(30,100),(100,300),(300,1000),(1000,3000),(3000,6000),(6000,10000),(10000,16000),(16000,24000)])
for nm, c in [("ours enh", o[:,2]), ("Adobe enh", a[:,2]), ("ours iso", o[:,3]), ("Adobe iso", a[:,3])]:
    print(f"  {nm:<10}", ltas(c))
