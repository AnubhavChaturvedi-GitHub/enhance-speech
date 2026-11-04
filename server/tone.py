"""Broad tone with no fixed target curve, because voices keep their own spectrum.

One zero-phase FIR per file: `tone_fir` analyses the file once and `dsp.apply_fir` applies the same taps
to every block, so a block split cannot change the result.

Two parts make up the one filter:

1. Rumble cut: about -44 dB at 0 Hz, -22 dB at 45 Hz, -2 dB at 80 Hz, flat from 100 Hz.

2. Low balance. The level of the octave around the voice's own median pitch is compared with the
   level of the first-formant zone, 350-700 Hz, and a low shelf whose gain falls as f0/f moves that
   difference to BALANCE_TARGET. It can lift or cut, and one number decides both. On the three reference
   outputs this balance reads 2.03, 2.29 and 3.17 dB, a narrow range for very different voices, while the
   input and the regenerated speech differ by 4 dB. Three voices can only prove so much, so it stays small
   and falls back when the file does not look like one voice.

Nothing here touches anything by more than a few dB, and every step falls back to the plain rumble
filter when the file has too little voice to judge: short clips, silence, noise and pure tones.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import firwin2, resample_poly

SR = 48000

# Low balance, chosen on the first two reference voices (2.03 and 2.29 dB). The third, held out, reads 3.17 dB.
BALANCE_TARGET = 2.2
LOW_ZONE = (0.71, 1.41)  # one octave around the median pitch, in multiples of it
REF_ZONE_HZ = (350.0, 700.0)  # the first-formant zone
MAX_LIFT_DB = 8.0
MAX_CUT_DB = 6.0
# A balance this far from the target is not a voice's tone: a phone line with no fundamental at all
# below, a hum or a sine above. The gain fades out over the last 4 dB.
OUT_OF_RANGE_DB = 18.0
# Above 250 Hz the two zones begin to overlap, so the balance stops saying much; the gain narrows to
# nothing at 450 Hz. Inside the two voices the balance was checked for pitches of 135 to 319 Hz.
OVERLAP_HZ = (250.0, 450.0)

# Confidence is voiced / (voiced + 1 s) of steady, loud voiced sound; below MIN_VOICED_S nothing happens.
# On excerpts the measured gain wanders by about 3.7 dB / sqrt(voiced seconds).
MIN_VOICED_S = 1.0
PITCH_RANGE = (70.0, 450.0)
# One static filter cannot serve two voices with different needs. A wide spread of pitch (log2 of the
# inter-quartile ratio) hints at that: 0.41 and 0.42 on the single voices, up to 0.64 on their 12 s
# excerpts, 0.63 and 0.75 on the two joined. The filter fades out across this range.
PITCH_SPREAD = (0.55, 0.85)

PROBE_LIMIT_S = 300  # analysed at most, spread evenly over longer files
NUMTAPS = 4095

_GRID = np.geomspace(20.0, 23900.0, 400)


@dataclass
class ToneInfo:
    """What the analysis found. For logs and tests; the taps are all the chain needs."""

    f0: float = 0.0
    voiced_s: float = 0.0
    confidence: float = 0.0
    balance_db: float = float("nan")
    low_gain_db: float = 0.0


def track_pitch(x: np.ndarray, sr: int = SR, fmin: float = 60.0, fmax: float = 500.0, hop_s: float = 0.01, win_s: float = 0.06, octave: float = 0.9):
    """Normalised autocorrelation pitch on an 8 kHz copy. Returns times (s), f0 (Hz), strength 0..1 and level (dB) per frame.

    f0 is the strongest lag's frequency whether voiced or not; callers threshold the strength. About 0.2 s per minute.
    """
    y = resample_poly(np.nan_to_num(np.asarray(x, dtype=np.float64)), 1, sr // 8000)
    fs = sr / (sr // 8000)
    size, hop = int(win_s * fs), int(hop_s * fs)
    if len(y) < size + hop:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
    nfft = 2 * size
    window = np.hanning(size)
    # Dividing by the window's own autocorrelation keeps long lags from being penalised.
    rw = np.fft.irfft(np.abs(np.fft.rfft(window, nfft)) ** 2)[:size]
    rw = np.maximum(rw / rw[0], 0.05)
    lo, hi = int(fs / fmax), min(int(fs / fmin), size - 2)
    starts = np.arange(0, len(y) - size, hop)
    f0, strength, level = np.zeros(len(starts)), np.zeros(len(starts)), np.zeros(len(starts))
    for first in range(0, len(starts), 2048):
        block = starts[first : first + 2048]
        frames = np.stack([y[i : i + size] for i in block])
        frames -= frames.mean(axis=1, keepdims=True)
        energy = (frames * frames).mean(axis=1)
        level[first : first + 2048] = 10 * np.log10(energy + 1e-14)
        r = np.fft.irfft(np.abs(np.fft.rfft(frames * window, nfft, axis=1)) ** 2, axis=1)[:, :size]
        r = r / (r[:, :1] + 1e-14) / rw
        seg = r[:, lo - 1 : hi + 2]
        is_peak = (seg[:, 1:-1] > seg[:, :-2]) & (seg[:, 1:-1] >= seg[:, 2:])
        peaks = np.where(is_peak, seg[:, 1:-1], -1.0)
        best = peaks.max(axis=1)
        # The shortest lag almost as strong as the best one: this is what stops period doubling.
        k = np.argmax(peaks >= octave * best[:, None], axis=1) + 1
        rows = np.arange(len(block))
        a, c, d = seg[rows, k - 1], seg[rows, k], seg[rows, k + 1]
        den = a - 2 * c + d
        shift = np.where(np.abs(den) > 1e-9, 0.5 * (a - d) / np.where(np.abs(den) > 1e-9, den, 1.0), 0.0)
        lag = lo - 1 + k + np.clip(shift, -1, 1)
        f0[first : first + 2048] = np.where(best > 0, fs / lag, 0.0)
        strength[first : first + 2048] = np.maximum(best, 0.0)
    return (starts + size / 2) / fs, f0, strength, level


def _probe(x: np.ndarray, limit_s: int) -> np.ndarray:
    """At most limit_s seconds of x, taken in evenly spread pieces, so a long file stays within RAM."""
    n, limit = len(x), limit_s * SR
    if n <= limit:
        return np.nan_to_num(np.asarray(x, dtype=np.float32))
    pieces, size = 10, limit // 10
    starts = np.linspace(0, n - size, pieces).astype(int)
    return np.concatenate([np.nan_to_num(np.asarray(x[s : s + size], dtype=np.float32)) for s in starts])


def _mean_power(x: np.ndarray, n_fft: int, hop: int, drop: float) -> np.ndarray:
    """Mean power spectrum over frames within `drop` dB of the file's loud level. A loud moment, 10 dB over it, would outweigh all the speech."""
    window = np.hanning(n_fft)
    power, count = np.zeros(n_fft // 2 + 1), 0
    starts = np.arange(0, len(x) - n_fft, hop)
    energy = np.array([(x[s : s + n_fft].astype(np.float64) ** 2).sum() for s in starts])
    level = 10 * np.log10(energy + 1e-14)
    loud = np.percentile(level, 95)
    starts = starts[(level > loud - drop) & (level < loud + 10.0)]
    for first in range(0, len(starts), 256):
        frames = np.stack([x[s : s + n_fft] for s in starts[first : first + 256]]).astype(np.float64)
        power += (np.abs(np.fft.rfft(frames * window, axis=1)) ** 2).sum(axis=0)
        count += len(frames)
    return power / max(1, count)


def _zone(power: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    return 10 * np.log10(power[(freqs >= lo) & (freqs < hi)].mean() + 1e-16)


def low_shape(freqs: np.ndarray, f0: float) -> np.ndarray:
    """1 at the pitch, f0/f above it, fading to 0 below 0.7 f0 so nothing under the voice is raised."""
    r = np.maximum(freqs, 1e-3) / f0
    above = np.where(r > 1, 1 / r, 1.0)
    t = np.clip(np.log(r / 0.4) / np.log(0.7 / 0.4), 0, 1)
    return above * (0.5 - 0.5 * np.cos(np.pi * t))


def _rumble_db(freqs: np.ndarray) -> np.ndarray:
    return np.interp(freqs, [0.0, 45.0, 80.0], [-60.0, -24.0, 0.0])


def _balance_after(power: np.ndarray, freqs: np.ndarray, f0: float, gain: float, shape: np.ndarray) -> float:
    p = power * 10 ** (gain * shape / 10)
    return _zone(p, freqs, LOW_ZONE[0] * f0, LOW_ZONE[1] * f0) - _zone(p, freqs, *REF_ZONE_HZ)


def low_gain(power: np.ndarray, freqs: np.ndarray, f0: float, target: float = BALANCE_TARGET) -> tuple[float, float]:
    """The shelf gain at f0 that brings the balance to `target`, and the balance before it. Solved on the spectrum, so no filtering is needed."""
    shape = low_shape(freqs, f0)
    before = _balance_after(power, freqs, f0, 0.0, shape)
    lo, hi = -MAX_CUT_DB, MAX_LIFT_DB
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if _balance_after(power, freqs, f0, mid, shape) > target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi) * float(np.clip((OUT_OF_RANGE_DB - abs(before - target)) / 4.0, 0, 1)), before


def _taps(curve_db: np.ndarray, grid: np.ndarray = _GRID, numtaps: int = NUMTAPS) -> np.ndarray:
    points = np.concatenate([[0.0], grid, [SR / 2]])
    gains = np.concatenate([[curve_db[0]], curve_db, [curve_db[-1]]])
    return firwin2(numtaps, points, 10 ** (gains / 20), fs=SR, window="hann").astype(np.float32)


def rumble_fir() -> np.ndarray:
    """The fallback: only the rumble cut."""
    return _taps(_rumble_db(_GRID))


def tone_fir(y: np.ndarray, side: np.ndarray, target: float = BALANCE_TARGET, probe_limit_s: int = PROBE_LIMIT_S) -> tuple[np.ndarray, ToneInfo]:
    """The file's one tone filter. y: the regenerated speech before any filtering. side: the isolated stem, time-aligned.

    Call once per file; both may be memmaps. Returns zero-phase taps and what was measured.
    """
    info = ToneInfo()
    n = min(len(y), len(side))
    if n < SR // 2:
        return rumble_fir(), info
    py, ps = _probe(y[:n], probe_limit_s), _probe(side[:n], probe_limit_s)
    if np.abs(ps).max() < 1e-4:
        return rumble_fir(), info
    t, f0, strength, level = track_pitch(ps)
    if len(t) < 3:
        return rumble_fir(), info
    loud = level > np.percentile(level, 95) - 15
    voiced = (strength >= 0.6) & loud
    steady = np.zeros(len(t), bool)
    steady[1:-1] = voiced[1:-1] & voiced[:-2] & voiced[2:] & (np.abs(f0[2:] - f0[:-2]) < 0.03 * f0[1:-1]) & (strength[1:-1] >= 0.7)
    info.voiced_s = float(steady.sum() * (t[1] - t[0]))
    if info.voiced_s < MIN_VOICED_S:
        return rumble_fir(), info
    info.f0 = float(np.median(f0[steady]))
    if not PITCH_RANGE[0] <= info.f0 <= PITCH_RANGE[1]:
        return rumble_fir(), info
    q25, q75 = np.percentile(f0[steady], [25, 75])
    spread = float(np.log2(q75 / q25))
    info.confidence = info.voiced_s / (info.voiced_s + 1.0) * float(np.clip((PITCH_SPREAD[1] - spread) / (PITCH_SPREAD[1] - PITCH_SPREAD[0]), 0, 1))
    if info.confidence <= 0:
        return rumble_fir(), info

    rumble = _rumble_db(_GRID)
    n_fft = 8192
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    # Judged as it will be heard, after the rumble cut.
    power = _mean_power(py, n_fft, 2048, 15.0) * 10 ** (np.interp(freqs, _GRID, rumble, left=rumble[0], right=0.0) / 10)
    gain, info.balance_db = low_gain(power, freqs, info.f0, target)
    gain *= float(np.clip((OVERLAP_HZ[1] - info.f0) / (OVERLAP_HZ[1] - OVERLAP_HZ[0]), 0, 1))
    info.low_gain_db = float(gain * info.confidence)
    return _taps(rumble + info.confidence * gain * low_shape(_GRID, info.f0)), info

