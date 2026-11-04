"""Signal processing around the models: fallback, the waveform and envelope blend, filtering and peaks."""
from __future__ import annotations

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.signal import fftconvolve

from .level import EXPORT_SPEECH_GAIN

SR = 48000
HYBRID_FFT, HYBRID_HOP = 1024, 240
# The recording's waveform below this crossover (Hz), the regenerated sound above it.
CROSSOVER = (4000.0, 8000.0)
# The most the regenerated envelope may lift the waveform, in dB. Sidon follows the input's level by now, so more than this is not speech.
MAX_LIFT_DB = 18.0
# Smoothing of both envelopes: 5 ms, and about half a harmonic's width at every pitch (2% of the frequency, at least 20 Hz).
ENVELOPE_S, ENVELOPE_Q, ENVELOPE_MIN_HZ = 0.005, 0.02, 20.0
# The envelopes are square-root power means, not power means, so a valley next to a peak still counts.
ENVELOPE_POWER = 0.5
# Where the guide asks for more than the waveform has, only this share of the lift in dB is taken. Cuts are taken in full.
LIFT_SHARE = 0.75
# How much of a cell's sound the separators kept, in dB. Below the first the cell is mostly noise, above the second it can be trusted.
KEPT_DB = (-9.0, -3.0)
# The same over 2 s, across the speech band. Above this a stretch of recording is clean throughout.
CLEAN_DB = (-3.0, -2.0)


def fallback_weight(primary: np.ndarray, safe: np.ndarray, speech_regions: list[tuple[int, int]]) -> np.ndarray:
    """How much of `safe` to use over time, 0 to 1: high only where speech exists and `primary` lost it.

    The separator gives the cleanest split when music is present, but under heavy noise and echo it
    can file the voice itself under "effects". A denoiser never does that, so it covers those spots.
    Music-only passages stay with the separator because no speech is detected there.
    """
    hop = 480
    window = 14400  # 300 ms
    # The running sum can leave a tiny negative rounding residue after loud sound stops.
    energy = lambda x: np.maximum(uniform_filter1d(x.astype(np.float64) ** 2, window), 0.0)[::hop] + 1e-12
    deficit = 10 * np.log10(energy(safe) / energy(primary))
    active = np.zeros(len(deficit))
    for start, end in speech_regions:
        active[start // hop : end // hop + 1] = 1
    if not active.any():
        return np.zeros(len(primary), np.float32)
    # When the separator struggles it struggles everywhere, so a file-wide loss switches over entirely.
    overall = np.clip((np.median(deficit[active > 0]) - 2) / 2, 0, 1)
    local = np.clip((deficit - 3) / 3, 0, 1)
    weight = np.maximum(local, overall) * active
    # Decide over about half a second, so the stem does not flicker between sources word by word.
    weight = uniform_filter1d(maximum_filter1d(weight, size=25), size=25)
    return np.interp(np.arange(len(primary)), np.arange(len(weight)) * hop, weight).astype(np.float32)


def _smoothstep(x: np.ndarray, a: float, b: float) -> np.ndarray:
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _envelope(power: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Smooths a power spectrogram. The width grows with frequency, so low harmonics stay resolved and high ones are not chased."""
    scales = [ENVELOPE_MIN_HZ, 40.0, 80.0, 160.0, 320.0]
    frames = ENVELOPE_S * SR / HYBRID_HOP
    root = (power + 1e-12) ** ENVELOPE_POWER
    stack = [gaussian_filter(root, (s / freqs[1], frames), truncate=3.0) for s in scales]
    position = np.interp(np.log(np.maximum(ENVELOPE_MIN_HZ, ENVELOPE_Q * freqs)), np.log(scales), np.arange(len(scales)))
    lower = np.clip(np.floor(position).astype(int), 0, len(scales) - 2)
    part = (position - lower)[:, None].astype(power.dtype)
    out = np.empty_like(power)
    for j in range(len(scales) - 1):
        rows = lower == j
        if rows.any():
            out[rows] = (1 - part[rows]) * stack[j][rows] + part[rows] * stack[j + 1][rows]
    return np.maximum(out, 1e-30) ** (1 / ENVELOPE_POWER)


def hybrid(wave: np.ndarray, guide: np.ndarray, mix: np.ndarray) -> np.ndarray:
    """The recording's own waveform shaped by the regenerated envelope below the crossover, the regenerated sound above it.

    A vocoder's waveform has nothing to do with the recording's: it shifts pitch a little, changes words now
    and then, and would comb-filter against the isolated stem at the Speech slider's middle positions. The
    envelope is what makes speech sound studio-recorded, so that is all that is taken from it, except where
    the waveform cannot be trusted: where the separators had to take away most of a cell's sound, the
    regenerated sound is used as it is. Heavy noise can make that the whole recording.
    """
    window = np.hanning(HYBRID_FFT).astype(np.float32)
    stft = lambda x: librosa.stft(x.astype(np.float32), n_fft=HYBRID_FFT, hop_length=HYBRID_HOP, window=window)
    W, G, M = stft(wave), stft(guide), stft(mix)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=HYBRID_FFT)
    PW, PG, PM = np.abs(W) ** 2, np.abs(G) ** 2, np.abs(M) ** 2
    ratio = (_envelope(PG, freqs) + 1e-12) / (_envelope(PW, freqs) + 1e-12)
    lift = np.minimum(np.where(ratio > 1, ratio ** (LIFT_SHARE / 2), np.sqrt(ratio)), 10 ** (MAX_LIFT_DB / 20))

    patch = (60.0 / freqs[1], 0.015 * SR / HYBRID_HOP)
    kept = 10 * np.log10((gaussian_filter(PW, patch, truncate=3.0) + 1e-12) / (gaussian_filter(PM, patch, truncate=3.0) + 1e-12))
    band = (freqs >= 200.0) & (freqs < 4000.0)
    region = int(2 * SR / HYBRID_HOP) | 1
    # The running sum can leave a tiny negative rounding residue after loud sound stops.
    sum_w = np.maximum(uniform_filter1d(PW[band].sum(axis=0).astype(np.float64), region, mode="nearest"), 0.0)
    sum_m = np.maximum(uniform_filter1d(PM[band].sum(axis=0).astype(np.float64), region, mode="nearest"), 0.0)
    clean = _smoothstep(10 * np.log10((sum_w + 1e-12) / (sum_m + 1e-12)), *CLEAN_DB).astype(np.float32)
    below = (0.5 + 0.5 * np.cos(np.pi * np.clip((freqs - CROSSOVER[0]) / (CROSSOVER[1] - CROSSOVER[0]), 0, 1)))[:, None].astype(np.float32)
    share = below * np.maximum(_smoothstep(kept, *KEPT_DB), clean[None, :])

    return librosa.istft(share * W * lift + (1 - share) * G, hop_length=HYBRID_HOP, window=window, length=len(wave)).astype(np.float32)


def apply_fir(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    return fftconvolve(x, taps, mode="same").astype(np.float32)


def limiter(x: np.ndarray, ceiling: float = 0.98, lookahead_ms: float = 5.0) -> np.ndarray:
    """Transparent peak limiter: only touches samples that would pass the ceiling."""
    peak = np.abs(x)
    if peak.max() <= ceiling:
        return x
    size = max(3, int(SR * lookahead_ms / 1000))
    gain = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    gain = minimum_filter1d(gain, size=size * 2 + 1)
    smooth = np.hanning(size * 2 + 1)
    gain = fftconvolve(gain, smooth / smooth.sum(), mode="same")
    return (x * np.minimum(gain, 1.0)).astype(np.float32)


def strength_gain(percent: float) -> float:
    """Slider position to linear gain for the background and music stems: 0 mutes, 100 is unity."""
    if percent <= 0:
        return 0.0
    if percent >= 100:
        return 1.0
    return float(10 ** (-36 * (1 - percent / 100) ** 2 / 20))


def download_gains(speech: float, speech_muted: bool, background: float, background_muted: bool, music: float, music_muted: bool, v1: bool) -> dict[str, float]:
    """Stem gains for a download, from the same slider positions the browser plays. In v2 the mono speech stems come down 3 dB, as in the reference downloads."""
    enhanced = speech / 100
    if v1:
        return {"original": 1 - enhanced, "enhanced": enhanced, "isolated": 0.0, "music": 0.0, "background": 0.0}
    return {
        "original": 0.0,
        "enhanced": 0.0 if speech_muted else enhanced * EXPORT_SPEECH_GAIN,
        "isolated": 0.0 if speech_muted else (1 - enhanced) * EXPORT_SPEECH_GAIN,
        "music": 0.0 if music_muted else strength_gain(music),
        "background": 0.0 if background_muted else strength_gain(background),
    }
