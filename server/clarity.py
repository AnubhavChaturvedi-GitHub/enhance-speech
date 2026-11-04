"""Harmonic clarity: lowers what lies between a voice's harmonics, as far as this file's own harmonics fall short of a clean voice's.

Each sample moves towards the mean of the two neighbouring pitch cycles, each lined up with this one first:
y(t) = x(t) + G * [0.5 * (x(t - Tb) + x(t + Tf)) - x(t)], where G is a zero-phase filter with strength g per frequency.
A steady harmonic adds up to x(t) itself and stays as it is, so the stem still blends with the isolated stem at the
Speech slider's middle positions. Halfway between harmonics the result is (1 - 2g) x(t).

How deep g goes is set once per file, from the gap between its contrast at each harmonic number and CLEAN_CURVE,
so a clear recording is left alone and a roomy one is not.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
from scipy.signal import fftconvolve, firwin2, medfilt, resample_poly

from .tone import track_pitch

SR = 48000
# 20 log10(harmonic / halfway to the next one) on steady voiced frames of the reference output, quadratic in log2(k).
# One curve fits all three reference voices within about 1 dB, at pitches of 158 to 247 Hz.
CLEAN_CURVE = (-0.74, -2.84, 33.35)
KMAX = 16
GAP_PER_G = 6.0  # dB of gap that asks for g = 1
GMAX = 0.5  # the deepest a three-point comb goes: a null halfway between harmonics
TOP_HZ = (2000.0, 3000.0)  # the comb fades out here: harmonics are not resolved this high in voiced speech and the lags are less sure
SECOND_G = 0.15  # the neighbours two cycles away, used only where the first pair has reached its limit, and only below SECOND_TOP_HZ
SECOND_TOP_HZ = (800.0, 1600.0)
MIN_FRAMES = 60  # steady voiced frames needed to believe the curve
QUIET_DB = -60.0  # below this the stem holds no voice to sharpen
# Speech stem against the input while someone speaks (level.speech_share). Below the first the separators took much of the input, so no comb.
NOISY_SHARE_DB = (-6.0, -3.0)
VOICE_STRENGTH = (0.70, 0.85)  # track_pitch's strength: no comb below the first, full comb above the second
PROBE_S = 300  # measured at most, spread evenly over longer files
NUMTAPS = 481
_HOP = 48  # lag search grid in samples, 1 ms, so each cycle is lined up on its own and real jitter stays


def _smoothstep(x: np.ndarray | float, a: float, b: float) -> np.ndarray:
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _steady(t: np.ndarray, f0: np.ndarray, strength: np.ndarray, level: np.ndarray) -> np.ndarray:
    """Frames of held, loud voice."""
    if len(t) < 5:
        return np.zeros(0, dtype=int)
    on = (strength >= 0.7) & (level > np.percentile(level, 95) - 15.0)
    ok = np.zeros(len(t), bool)
    ok[1:-1] = on[1:-1] & on[:-2] & on[2:] & (np.abs(f0[2:] - f0[:-2]) < 0.03 * f0[1:-1])
    return np.flatnonzero(ok)


def _refine(magnitude: np.ndarray, f: float, df: float) -> float:
    """The pitch near f whose first eight harmonics land on the most energy."""
    candidates = f * (1 + np.linspace(-0.012, 0.012, 13))
    idx = np.minimum(np.round(np.outer(candidates, np.arange(1, 9)) / df).astype(int), len(magnitude) - 1)
    return float(candidates[np.argmax(np.log(magnitude[idx] + 1e-9).sum(axis=1))])


def _harmonic_curve(x: np.ndarray, track: tuple, maxframes: int = 1500, nfft: int = 16384) -> tuple[np.ndarray, int, float]:
    """Mean of 20 log10(harmonic / halfway to the next) per harmonic number on steady voiced frames. Returns the curve, the frames used and the median pitch.

    A four-period Hann window: a clean harmonic's own leakage is nil at the halfway point.
    """
    t, f0, strength, level = track
    x = np.asarray(x, dtype=np.float64)
    idx = _steady(t, f0, strength, level)[::2]
    if len(idx) > maxframes:
        idx = idx[np.linspace(0, len(idx) - 1, maxframes).astype(int)]
    df, rows, used = SR / nfft, [], []
    ks = np.arange(1, KMAX + 1)
    for i in idx:
        f = float(f0[i])
        center, size = int(t[i] * SR), int(4 * SR / f) // 2 * 2
        if center - size // 2 < 0 or center + size // 2 >= len(x):
            continue
        magnitude = np.abs(np.fft.rfft(x[center - size // 2 : center + size // 2] * np.hanning(size), nfft))
        f = _refine(magnitude, f, df)
        peak, valley = np.round(ks * f / df).astype(int), np.round((ks + 0.5) * f / df).astype(int)
        keep = valley < len(magnitude)
        row = np.full(KMAX, np.nan)
        row[keep] = 20 * np.log10((magnitude[peak[keep]] + 1e-9) / (magnitude[valley[keep]] + 1e-9))
        rows.append(row)
        used.append(f)
    if not rows:
        return np.full(KMAX, np.nan), 0, 0.0
    return np.nanmean(np.array(rows), axis=0), len(rows), float(np.median(used))


def _taps(g: np.ndarray, f0: float, top: tuple[float, float]) -> np.ndarray:
    harmonics = f0 * np.arange(1, len(g) + 1)
    freqs = np.r_[0.0, np.linspace(harmonics[0], 8000.0, 160), SR / 2]
    fade = 0.5 + 0.5 * np.cos(np.pi * np.clip((freqs - top[0]) / (top[1] - top[0]), 0, 1))
    return firwin2(NUMTAPS, freqs, np.interp(freqs, harmonics, g, left=g[0], right=g[-1]) * fade, fs=SR, window="hann").astype(np.float32)


def _identity(reason: str, **extra) -> dict:
    return {"taps": None, "taps2": None, "g": np.zeros(KMAX), "f0": 0.0, "frames": 0, "top_db": 0.0, "reason": reason} | extra


def _plan_from_curve(curve: np.ndarray, frames: int, f0: float, top_db: float, trust: float) -> dict:
    if frames < MIN_FRAMES or f0 <= 0:
        return _identity("too little steady voice", frames=frames, f0=f0)
    gap = np.polyval(CLEAN_CURVE, np.log2(np.arange(1, len(curve) + 1))) - curve
    gap = np.where(np.isnan(gap), 0.0, gap)
    padded = np.r_[gap[0], gap, gap[-1]]
    gap = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    # With few frames the curve wanders, so it counts for less.
    g = np.clip(gap / GAP_PER_G, 0.0, GMAX) * (frames / (frames + 50.0)) * trust
    if g.max() < 0.02:
        return _identity("already as clear", frames=frames, f0=f0, gap=gap)
    g2 = SECOND_G * _smoothstep(g, 0.3, 0.45)
    taps2 = _taps(g2, f0, SECOND_TOP_HZ) if g2.max() > 0.01 else None
    return {"taps": _taps(g, f0, TOP_HZ), "taps2": taps2, "g": g, "f0": f0, "frames": frames, "top_db": top_db, "gap": gap, "reason": "measured"}


def _probe(x: np.ndarray) -> np.ndarray:
    n, limit = len(x), PROBE_S * SR
    if n <= limit:
        return np.nan_to_num(np.asarray(x, dtype=np.float32))
    size = limit // 10
    return np.concatenate([np.nan_to_num(np.asarray(x[int(s) : int(s) + size], dtype=np.float32)) for s in np.linspace(0, n - size, 10)])


def estimate_clarity(stem: np.ndarray, isolated: np.ndarray, share_db: float = 0.0) -> dict:
    """Measures once per file how deep the comb goes. Accepts memmaps. The pitch is read from the isolated stem; share_db is level.speech_share's second value."""
    n = min(len(stem), len(isolated))
    if n < SR // 2:
        return _identity("too short")
    trust = float(_smoothstep(np.array(share_db), *NOISY_SHARE_DB))
    if trust <= 0:
        return _identity("too noisy")
    y, side = _probe(stem[:n]), _probe(isolated[:n])
    if np.abs(side).max() < 1e-4:
        return _identity("silence")
    track = track_pitch(side)
    if len(track[0]) < 5:
        return _identity("too short")
    top = float(np.percentile(track[3], 95))
    if top < QUIET_DB:
        return _identity("no voice to sharpen")
    curve, frames, f0 = _harmonic_curve(y, track)
    return _plan_from_curve(curve, frames, f0, top, trust)


def _voiced_track(t: np.ndarray, f0: np.ndarray, strength: np.ndarray, level: np.ndarray, top_db: float) -> tuple[np.ndarray, np.ndarray]:
    """Log period per frame, and a voiced weight that stays 0 outside held voice. top_db: the file's loud level, measured once."""
    f = medfilt(np.maximum(f0, 1.0), 5)
    v = _smoothstep(strength, *VOICE_STRENGTH) * _smoothstep(level, top_db - 30.0, top_db - 24.0)
    jump = np.abs(np.log2(np.r_[f[1:], f[-1]] / f)) > 0.25
    v = np.where(jump | np.r_[False, jump[:-1]], 0.0, v)
    edges = np.diff(np.r_[0, (v > 0).astype(int), 0])
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if end - start < 5:
            v[start:end] = 0.0
    return np.log(SR / f), gaussian_filter1d(v, 1.5)


def _at(x: np.ndarray, position: np.ndarray) -> np.ndarray:
    i = np.clip(np.floor(position).astype(np.int64), 0, len(x) - 2)
    a = np.clip(position - i, 0.0, 1.0).astype(np.float32)
    return x[i] * (1 - a) + x[i + 1] * a


def _lags(x: np.ndarray, times: np.ndarray, log_period: np.ndarray, sign: int, period: float, span: float = 0.04, steps: int = 17, win: float = 1.25) -> tuple[np.ndarray, np.ndarray]:
    """Per ms, the lag that best lines the cycle one period away up with this one, searched on a 16 kHz copy. period: the file's median period, in samples."""
    y = resample_poly(np.asarray(x, dtype=np.float64), 1, 3).astype(np.float32)
    n = len(y)
    idx = np.arange(n, dtype=np.float64)
    lag = np.exp(np.interp(idx / 16000.0, times, log_period)) / 3.0
    window = max(3, int(win * period / 3.0)) | 1
    frames = np.arange(0, n, _HOP // 3)
    grid = 1 + np.linspace(-span, span, steps)
    energy = np.maximum(uniform_filter1d(y * y, window, mode="nearest")[frames], 0.0)
    score = np.zeros((steps, len(frames)), np.float32)
    for j, m in enumerate(grid):
        z = _at(y, idx + sign * lag * m)
        product = uniform_filter1d(y * z, window, mode="nearest")[frames]
        other = np.maximum(uniform_filter1d(z * z, window, mode="nearest")[frames], 0.0)
        score[j] = product / np.sqrt(np.maximum(energy * other, 1e-14))
    best = np.clip(np.argmax(score, axis=0), 1, steps - 2)
    rows = np.arange(len(frames))
    a, c, b = score[best - 1, rows], score[best, rows], score[best + 1, rows]
    den = a - 2 * c + b
    shift = np.where(np.abs(den) > 1e-7, 0.5 * (a - b) / np.where(np.abs(den) > 1e-7, den, 1.0), 0.0)
    m = median_filter(grid[best] + np.clip(shift, -1, 1) * (grid[1] - grid[0]), size=5, mode="nearest")
    frame_times = frames / 16000.0
    return frame_times, np.exp(np.interp(frame_times, times, log_period)) * m


def apply_clarity(x: np.ndarray, isolated: np.ndarray, plan: dict) -> np.ndarray:
    """Applies a plan to one block. Matches a whole-file run when the block starts on the 480-sample grid and carries 2 s of margin."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float32))
    side = np.nan_to_num(np.asarray(isolated, dtype=np.float32))
    n = len(x)
    if plan["taps"] is None or n < SR // 4:
        return x
    t, f0, strength, level = track_pitch(side)
    if len(t) < 5:
        return x
    log_period, v = _voiced_track(t, f0, strength, level, plan["top_db"])
    if not (v > 0).any():
        return x
    period = SR / plan["f0"]
    times, lag_on = _lags(side, t, log_period, +1, period)
    _, lag_back = _lags(side, t, log_period, -1, period)
    idx = np.arange(n, dtype=np.float64)
    seconds = idx / SR
    forward, back = idx + np.interp(seconds, times, lag_on), idx - np.interp(seconds, times, lag_back)
    gain = np.interp(seconds, t, v).astype(np.float32)
    y = x + gain * fftconvolve(0.5 * (_at(x, back) + _at(x, forward)) - x, plan["taps"], mode="same").astype(np.float32)
    if plan["taps2"] is not None:
        forward2, back2 = forward + np.interp(forward / SR, times, lag_on), back - np.interp(back / SR, times, lag_back)
        y = y + gain * fftconvolve(0.5 * (_at(x, back2) + _at(x, forward2)) - x, plan["taps2"], mode="same").astype(np.float32)
    return y.astype(np.float32)
