"""Limits the stem by the recording: its short-time band envelope may not rise above the recording's, level-matched once per file, from 1 kHz up.

Regenerated speech puts energy in places the recording does not have it, and the reference output does not. This only cuts, and leaves
alone any band the recording does not hold, such as the highs above a codec's lowpass, and any file where the recording is mostly not the voice.

    plan = limit_plan(stem, mono, strength_for(whole))   # once per file, reads in chunks
    y = limit_block(stem[lo:hi], mono[lo:hi], plan)      # per block; starts on the 120-sample grid, at least 50 ms of margin
"""
from __future__ import annotations

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d

SR = 48000
FFT, HOP = 512, 120
WINDOW_S = 0.040
FMIN = 1000.0  # below this the blend already is the recording's waveform, and limiting there only hurt
MAX_CUT_DB = 12.0  # one wrong level offset cannot do more than this
NOISE_PERCENTILE = 10
# Speech stem against the input over the whole file. At or below the first the recording's envelope is not the voice's and the limit is off; at or above the second it is fully on.
SHARE_DB = (-8.0, -3.0)
GUARD_DB = 12.0  # a band this far above its 1-4 kHz level offset is not in the recording at all, so the regenerated sound stays
_centers = 125.0 * 2 ** (np.arange(0, 24) / 3)
CENTERS = _centers[_centers < 20000.0]
_FREQS = librosa.fft_frequencies(sr=SR, n_fft=FFT)
_BANDS = [(_FREQS >= c / 2 ** (1 / 6)) & (_FREQS < c * 2 ** (1 / 6)) for c in CENTERS]
_BANDS = [b if b.any() else np.abs(_FREQS - c) == np.abs(_FREQS - c).min() for b, c in zip(_BANDS, CENTERS)]
_WINDOW = np.hanning(FFT).astype(np.float32)
_log_freqs, _log_centers = np.log2(np.maximum(_FREQS, 1.0)), np.log2(CENTERS)
_LOWER = np.clip(np.searchsorted(_log_centers, _log_freqs) - 1, 0, len(_log_centers) - 2)
_PART = np.clip((_log_freqs - _log_centers[_LOWER]) / (_log_centers[_LOWER + 1] - _log_centers[_LOWER]), 0, 1)[:, None].astype(np.float32)


def _band_power(x: np.ndarray, center: bool = True) -> tuple[np.ndarray, np.ndarray]:
    spectrum = librosa.stft(np.nan_to_num(np.asarray(x, dtype=np.float32)), n_fft=FFT, hop_length=HOP, window=_WINDOW, center=center)
    power = np.abs(spectrum) ** 2
    return spectrum, np.stack([power[b].mean(axis=0) for b in _BANDS]).astype(np.float64)


def strength_for(whole_share_db: float) -> float:
    """How hard to limit, 0 to 1, from how much of the input the speech stem kept over the whole file (level.speech_share's first value)."""
    return float(np.clip((whole_share_db - SHARE_DB[0]) / (SHARE_DB[1] - SHARE_DB[0]), 0, 1))


def limit_plan(stem: np.ndarray, recording: np.ndarray, strength: float, seconds_max: float = 600.0, chunk_s: float = 30.0) -> dict:
    """The stem's level over the recording per band, on the file's loud frames. Reads at most seconds_max.

    The offset is the median over 250 ms pieces, so stretches where the recording holds something besides the voice do not set it.
    """
    n = min(len(stem), len(recording))
    identity = {"offset": np.full(len(CENTERS), np.inf), "active": False, "strength": 0.0}
    if n < FFT * 4 or strength <= 0:
        return identity
    step, limit = int(chunk_s * SR), int(seconds_max * SR)
    spans = [(a, min(n, a + step)) for a in range(0, n, step)] if n <= limit else [(int(a), int(a) + step) for a in np.linspace(0, n - step, max(1, limit // step))]
    piece = int(0.25 * SR / HOP)
    stems, recordings = [], []
    for a, b in spans:
        if b - a < FFT:
            continue
        _, ps = _band_power(stem[a:b], center=False)
        _, pr = _band_power(recording[a:b], center=False)
        whole_pieces = (ps.shape[1] // piece) * piece
        stems.append(ps[:, :whole_pieces] if whole_pieces else ps)
        recordings.append(pr[:, :whole_pieces] if whole_pieces else pr)
    ps, pr = np.concatenate(stems, axis=1), np.concatenate(recordings, axis=1)
    total = ps.sum(axis=0)
    if total.max() < 1e-10 or pr.sum() < 1e-10:
        return identity
    # The second test holds when speech fills under 30% of the file: gaps would otherwise count as the voice and the limit would cut it.
    loud = (total > np.percentile(total, 70)) & (total > np.percentile(total, 99.5) * 1e-3)
    pieces = ps.shape[1] // piece
    if pieces == 0:
        offset = ps[:, loud].sum(axis=1) / (pr[:, loud].sum(axis=1) + 1e-14)
    else:
        mask = loud[: pieces * piece].reshape(pieces, piece)
        use = mask.sum(axis=1) >= piece // 4
        if not use.any():
            return identity
        # The recording's steady noise comes off first, so the offset is the stem against the voice, and noise in the recording only loosens the limit.
        smooth = max(1, int(round(WINDOW_S * SR / HOP)))
        noise = np.percentile(uniform_filter1d(pr, smooth, axis=1, mode="nearest"), NOISE_PERCENTILE, axis=1)
        count = mask[use].sum(axis=1)
        offset = np.zeros(len(CENTERS))
        for i in range(len(CENTERS)):
            in_stem = (ps[i, : pieces * piece].reshape(pieces, piece) * mask)[use].sum(axis=1)
            in_recording = (pr[i, : pieces * piece].reshape(pieces, piece) * mask)[use].sum(axis=1)
            voice = np.maximum(in_recording - count * max(noise[i], 0.0), 0.1 * in_recording)
            offset[i] = np.median(in_stem / (voice + 1e-14))
    db = 10 * np.log10(offset + 1e-14)
    held = db <= np.median(db[(CENTERS >= 1000.0) & (CENTERS < 4000.0)]) + GUARD_DB
    return {"offset": np.where(held, offset, np.inf), "active": True, "strength": strength}


def limit_block(stem: np.ndarray, recording: np.ndarray, plan: dict) -> np.ndarray:
    """Applies a plan to one block."""
    stem = np.nan_to_num(np.asarray(stem, dtype=np.float32))
    if not plan["active"] or len(stem) < FFT:
        return stem
    spectrum, ps = _band_power(stem)
    _, pr = _band_power(recording)
    k = min(ps.shape[1], pr.shape[1])
    size = max(1, int(round(WINDOW_S * SR / HOP)))
    # The running sums can leave a tiny negative rounding residue after loud sound stops.
    stem_level = np.maximum(uniform_filter1d(ps[:, :k], size, axis=1, mode="nearest"), 0.0)
    recording_level = np.maximum(uniform_filter1d(pr[:, :k], size, axis=1, mode="nearest"), 0.0)
    ceiling = recording_level * np.where(np.isfinite(plan["offset"]), plan["offset"], 1.0)[:, None]
    gain = np.maximum(np.minimum(1.0, np.sqrt(ceiling / (stem_level + 1e-14))), 10 ** (-MAX_CUT_DB / 20))
    gain[(CENTERS < FMIN) | ~np.isfinite(plan["offset"])] = 1.0
    gain = gain ** plan["strength"]
    per_bin = (1 - _PART) * gain[_LOWER].astype(np.float32) + _PART * gain[_LOWER + 1].astype(np.float32)
    return librosa.istft(spectrum[:, :k] * per_bin, hop_length=HOP, window=_WINDOW, length=len(stem)).astype(np.float32)
