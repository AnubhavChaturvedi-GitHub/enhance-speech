"""Compares an enhanced recording with a reference enhanced recording of the same input, sample for sample.

usage: scorecard.py <ours> <reference> [<isolated stem>]

Each file is a mono stem, a stereo download or an 8-channel merged.mp4, whose enhanced channel is used. ffmpeg folds
a stereo file with identical channels back to the mono stem's level, so a download reads at its stem's loudness.
Both files must line up in time. Needs Praat for pitch and voicing: uv pip install praat-parselmouth pyloudnorm
Judge on more than one voice: a curve fitted on one voice and scored on that voice proves nothing.
"""
import subprocess, sys
import numpy as np, librosa, parselmouth, pyloudnorm
from parselmouth.praat import call
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

SR = 48000
BANDS = [(100, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]
CENTERS = np.array([125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500], float)
db = lambda x: 20 * np.log10(np.abs(x) + 1e-12)


def load(path):
    channels = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels", "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout.split()[0])
    if channels not in (1, 2, 8):
        sys.exit(f"{path} has {channels} channels; expected a mono stem, a stereo download or a merged.mp4")
    fold = ["-af", "pan=mono|c0=c2"] if channels == 8 else ["-ac", "1"]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0", *fold, "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="float32").copy()


def pitch(x, step=0.005):
    p = parselmouth.Sound(x.astype(np.float64), SR).to_pitch_ac(time_step=step, pitch_floor=60, pitch_ceiling=500)
    return p.xs(), p.selected_array["frequency"].copy()


def cents(a, b):
    fa, fb = pitch(a)[1], pitch(b)[1]
    n = min(len(fa), len(fb))
    both = (fa[:n] > 0) & (fb[:n] > 0)
    return float(np.median(1200 * np.abs(np.log2(fb[:n][both] / fa[:n][both]))))


def active_gain(x, ref):
    fx, fr = (librosa.feature.rms(y=s, frame_length=2400, hop_length=480)[0] for s in (x, ref))
    n = min(len(fx), len(fr))
    keep = fr[:n] > np.percentile(fr[:n], 60)
    return float(np.sqrt((fr[:n][keep] ** 2).mean() / max(1e-14, (fx[:n][keep] ** 2).mean())))


def logmel(x):
    return 10 * np.log10(librosa.feature.melspectrogram(y=x, sr=SR, n_fft=2048, hop_length=240, n_mels=64, fmin=60, fmax=20000) + 1e-10)


def classes(ref, hop=240):
    """Voiced, unvoiced and gap frames of the reference."""
    n = len(ref) // hop + 1
    t, f0 = pitch(ref, hop / SR)
    voiced = np.interp(np.arange(n) * hop / SR, t, (f0 > 0).astype(float)) > 0.5
    rms = librosa.feature.rms(y=ref, frame_length=960, hop_length=hop)[0]
    level = db(np.pad(rms, (0, max(0, n - len(rms))))[:n])
    loud = level > np.percentile(level, 95) - 35
    return voiced, ~voiced & loud, ~voiced & ~loud


def band_levels(x, mask):
    power = np.abs(librosa.stft(x, n_fft=4096, hop_length=1024)) ** 2
    k = min(power.shape[1], len(mask))
    mean = power[:, :k][:, mask[:k]].mean(axis=1)
    f = librosa.fft_frequencies(sr=SR, n_fft=4096)
    return np.array([10 * np.log10(mean[(f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))].mean() + 1e-14) for c in CENTERS])


def speech_masks(ref):
    power = np.abs(librosa.stft(ref, n_fft=4096, hop_length=1024)) ** 2
    f = librosa.fft_frequencies(sr=SR, n_fft=4096)
    level = 10 * np.log10(power.sum(axis=0) + 1e-14)
    t, f0 = pitch(ref, 1024 / SR)
    voiced = np.interp(np.arange(power.shape[1]) * 1024 / SR, t, (f0 > 0).astype(float)) > 0.5
    high, low = power[(f >= 3500) & (f < 12000)].sum(axis=0), power[(f >= 200) & (f < 2000)].sum(axis=0)
    return voiced & (level > np.percentile(level, 95) - 15), ~voiced & (level > np.percentile(level, 95) - 30) & (high > low * 0.5)


def voice_stats(x):
    s = parselmouth.Sound(x.astype(np.float64), SR)
    h = s.to_harmonicity_cc(time_step=0.01, minimum_pitch=75).values
    cepstrogram = call(s, "To PowerCepstrogram", 60.0, 0.002, 5000.0, 50.0)
    return float(h[h > -100].mean()), float(call(cepstrogram, "Get CPPS", False, 0.01, 0.001, 60.0, 330.0, 0.05, "Parabolic", 0.001, 0.05, "Straight", "Robust"))


def harmonic_contrast(x):
    """Peak over valley, in dB, between the harmonics of steady voiced frames."""
    t, f0 = pitch(x, 0.01)
    rms = librosa.feature.rms(y=x, frame_length=960, hop_length=480)[0]
    floor = np.percentile(rms, 95) * 0.1
    results, nfft = [[] for _ in BANDS], 16384
    for i in range(1, len(f0) - 1):
        f = f0[i]
        if f <= 0 or f0[i - 1] <= 0 or f0[i + 1] <= 0 or abs(f0[i + 1] - f0[i - 1]) / f > 0.015:
            continue
        c, size = int(t[i] * SR), int(4 * SR / f) // 2 * 2
        if c - size // 2 < 0 or c + size // 2 >= len(x) or rms[min(c // 480, len(rms) - 1)] < floor:
            continue
        m = np.abs(np.fft.rfft(x[c - size // 2 : c + size // 2] * np.hanning(size), nfft))
        for b, (lo, hi) in enumerate(BANDS):
            ks = np.arange(max(1, int(np.ceil(lo / f))), int(hi / f))
            if len(ks):
                results[b].append(db((m[np.round(ks * f * nfft / SR).astype(int)] + 1e-9) / (m[np.round((ks + 0.5) * f * nfft / SR).astype(int)] + 1e-9)).mean())
    return [float(np.mean(v)) if v else float("nan") for v in results]


def waveform_match(x, ref, window=0.1):
    """Correlation with the reference per band, best lag within 2 ms, weighted by the reference's energy."""
    out, size, reach = [], int(window * SR), int(0.002 * SR)
    for lo, hi in [(200, 1000), (1000, 2000), (2000, 4000)]:
        s = butter(4, [lo, hi], btype="band", fs=SR, output="sos")
        px, pr = sosfiltfilt(s, x), sosfiltfilt(s, ref)
        values, weights = [], []
        for at in range(0, len(px) - size, size):
            a, b = px[at : at + size], pr[at : at + size]
            ea, eb = (a * a).sum(), (b * b).sum()
            if eb < 1e-8 or ea < 1e-12:
                continue
            cc = np.fft.irfft(np.fft.rfft(a, 2 * size) * np.conj(np.fft.rfft(b, 2 * size)))
            values.append(np.max(np.r_[cc[:reach], cc[-reach:]]) / np.sqrt(ea * eb))
            weights.append(eb)
        out.append(float(np.average(values, weights=weights)))
    return out


def below(x):
    """Percent of 20 ms frames below -30/-40/-50/-60 dB, relative to the file's loud speech."""
    level = db(librosa.feature.rms(y=x, frame_length=960, hop_length=480)[0])
    level -= np.percentile(level, 95)
    return [round(float((level < v).mean() * 100), 1) for v in (-30, -40, -50, -60)]


def before_words(x, ref):
    """Median level relative to the word at 100/60/40/20/10/5 ms before a word starts, 500 Hz-4 kHz. Word starts come from the reference."""
    s = butter(4, [500, 4000], btype="band", fs=SR, output="sos")
    envelope = lambda z: db(np.sqrt(uniform_filter1d(sosfiltfilt(s, z) ** 2, 240)[::48] + 1e-14))
    r, e = envelope(ref), envelope(x)
    starts, i = [], 400
    while i < len(r) - 200:
        if r[i : i + 60].mean() > np.percentile(r, 80) - 6 and r[i - 150 : i - 40].max() < r[i : i + 60].mean() - 15:
            starts.append(i)
            i += 300
        else:
            i += 1
    return np.round(np.median([[e[j - t] - e[j : j + 60].mean() for t in (100, 60, 40, 20, 10, 5)] for j in starts], axis=0), 1)


def report(name, y, ref, isolated):
    y = y[: len(ref)]
    gained = y * active_gain(y, ref)
    A, Y = logmel(ref), logmel(gained)
    n = min(A.shape[1], Y.shape[1])
    voiced, unvoiced, gap = (m[:n] for m in classes(ref))
    distance = lambda m: float(np.abs(Y[:, :n][:, m] - A[:, :n][:, m]).mean())
    vowels, fricatives = speech_masks(ref)
    mid = (CENTERS >= 300) & (CENTERS <= 3000)
    offset = (band_levels(ref, vowels) - band_levels(gained, vowels))[mid].mean()
    vowel_error = band_levels(gained, vowels) + offset - band_levels(ref, vowels)
    fricative_error = band_levels(gained, fricatives) + offset - band_levels(ref, fricatives)
    hnr, cpps = voice_stats(y)
    print(f"{name}\n  distance {distance(voiced | unvoiced):.2f} (voiced {distance(voiced):.2f}, unvoiced {distance(unvoiced):.2f}, gaps {distance(gap):.2f})")
    print(f"  tone error, rms dB: vowels {np.sqrt((vowel_error ** 2).mean()):.1f}, fricatives {np.sqrt((fricative_error ** 2).mean()):.1f}")
    print(f"  pitch against the reference {cents(ref, y):.1f} cents" + (f", against the recording {cents(isolated, y):.1f} cents" if isolated is not None else ""))
    print(f"  HNR {hnr:.1f}, CPPS {cpps:.2f}, harmonic contrast {' / '.join(f'{c:.1f}' for c in harmonic_contrast(y))}")
    print(f"  waveform match 200 Hz-1 kHz / 1-2 kHz / 2-4 kHz: {' / '.join(f'{c:.2f}' for c in waveform_match(gained, ref))}")
    print(f"  loudness {pyloudnorm.Meter(SR).integrated_loudness(y.astype(np.float64)):.1f} LUFS, peak {db(np.abs(y).max()):.1f} dBFS")
    print(f"  time below -30/-40/-50/-60 dB: {below(y)}")
    print(f"  before a word, 100/60/40/20/10/5 ms: {before_words(y, ref)}")


if __name__ == "__main__":
    ours, reference = load(sys.argv[1]), load(sys.argv[2])
    isolated = load(sys.argv[3]) if len(sys.argv) > 3 else None
    report("Reference", reference, reference, isolated)
    report("Ours", ours, reference, isolated)
