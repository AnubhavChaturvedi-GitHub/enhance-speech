"""Output level: a fixed loudness for the enhanced stem, and the -3 dB a download gives the mono speech stems.

Measured on reference enhanced tracks, see bench/README.md. The enhanced stem sits at the same loudness
whatever the input level (-21.2, -20.9 and -20.4 LUFS on three voices, for inputs from -13.8 to -24.5 LUFS). A downloaded mix then
scales the mono speech stems by 1/sqrt(2) and leaves the stereo stems alone. The loudness step runs once
per file, so a block split cannot change the gain.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import sosfilt

from . import gate

SR = 48000

# Enhanced stem, on the same scale as ffmpeg's ebur128. Midpoint of the two reference stems.
TARGET_LUFS = -20.8
# Mono speech stems in a downloaded stereo mix. Measured as 0.7071 on two separate reference files.
EXPORT_SPEECH_GAIN = 2 ** -0.5

# ITU-R BS.1770 K-weighting at 48 kHz.
_K_WEIGHT = np.array([
    [1.53512485958697, -2.69169618940638, 1.19839281085285, 1.0, -1.69065929318241, 0.73248077421585],
    [1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621],
])
_HOP = SR // 10
_HOPS_PER_READ = 600  # 60 s
# A frame is active within this much of the speech stem's loudest level.
ACTIVE_DB = 20.0
# 100 ms of a normal recording stays under this much over its median voiced frame, on ten real recordings.
RECORDING_OVER_DB = 18.0


def _finite(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def _hop_powers(x: np.ndarray) -> np.ndarray:
    """K-weighted mean power per 100 ms of a mono signal, reading x a minute at a time so a memory-mapped file stays cheap."""
    state = np.zeros((2, 2))
    out = []
    usable = len(x) // _HOP * _HOP
    for start in range(0, usable, _HOP * _HOPS_PER_READ):
        chunk = _finite(x[start : min(usable, start + _HOP * _HOPS_PER_READ)])
        y, state = sosfilt(_K_WEIGHT, chunk, zi=state)
        out.append((y * y).reshape(-1, _HOP).mean(axis=1))
    return np.concatenate(out) if out else np.zeros(0)


def integrated_lufs(x: np.ndarray, skip: np.ndarray | None = None) -> float:
    """BS.1770 integrated loudness of a mono signal at 48 kHz; -inf for silence. `skip` marks 100 ms hops to leave out.

    A clip shorter than one 400 ms gating block has no gated value, so it is measured as a single block.
    """
    powers = _hop_powers(x)
    if skip is not None and skip.any():
        keep = np.ones(len(powers), bool)
        keep[: len(skip)] = ~skip[: len(powers)]
        if keep.sum() >= 4:
            powers = powers[keep]
    if len(powers) < 4:
        if len(x) == 0:
            return float("-inf")
        y = sosfilt(_K_WEIGHT, _finite(x))
        level = float((y * y).mean())
        return -0.691 + 10 * np.log10(level) if level > 1e-13 else float("-inf")
    blocks = np.convolve(powers, np.full(4, 0.25), mode="valid")
    loudness = -0.691 + 10 * np.log10(blocks + 1e-30)
    above = loudness > -70
    if not above.any():
        return float("-inf")
    threshold = -0.691 + 10 * np.log10(blocks[above].mean()) - 10
    kept = above & (loudness > threshold)
    return -0.691 + 10 * np.log10(blocks[kept].mean())


def _ratio(wanted: float, have: float) -> float:
    return float(np.sqrt(wanted / have)) if have > 1e-20 else 1.0


def follow_input(restored: np.ndarray, reference: np.ndarray, seconds: float = 4.0) -> np.ndarray:
    """Puts one block of Sidon at the level of the speech it was made from, over a few seconds at a time.

    Sidon's features are normalised per block, so its output level ignores the input: each block comes out
    near -22 dB RMS. Left alone that is a block-wise level control. It steps at seams, moves with where the
    seams fall, and blows residue up to speech level when a block holds no voice. Run this per block,
    before the blocks are cross-faded together and before anything multiplies Sidon with the waveform.
    """
    if len(restored) < 4 * _HOP:
        ratio = _ratio(float((_finite(reference) ** 2).mean()), float((_finite(restored) ** 2).mean()))
        return (_finite(restored) * ratio).astype(np.float32)
    # A loud moment comes out of Sidon far less loud than it went in. Left to fill both averages, it would lift the speech for seconds around it.
    ps, pr = _held(restored), _held(reference)
    size = max(1, int(seconds * SR / _HOP))
    # The running sums can leave a tiny negative rounding residue after loud sound stops.
    smooth = lambda p: np.maximum(uniform_filter1d(uniform_filter1d(p, size, mode="nearest"), size, mode="nearest"), 0.0)
    ms, mr = smooth(ps), smooth(pr)
    # Where neither signal holds speech the ratio means nothing, so it falls back to the block's overall ratio.
    floor = float(np.percentile(mr, 95)) * 1e-3 + 1e-14
    overall = (float(pr.mean()) + 1e-14) / (float(ps.mean()) + 1e-14)
    gain = np.sqrt((mr + floor) / (ms + floor / overall))
    centres = (np.arange(len(gain)) + 0.5) * _HOP
    return (_finite(restored) * np.interp(np.arange(len(restored)), centres, gain)).astype(np.float32)


def _held(x: np.ndarray) -> np.ndarray:
    """x's 100 ms powers, held to a loud moment's level over its voice. The same powers on a file without loud moments."""
    powers, voice = _hop_powers(x), gate.speech_level(x)
    return powers if voice is None else np.minimum(powers, 10 ** ((voice + gate.LOUD_TO) / 10))


def speech_share(speech: np.ndarray, mix: np.ndarray) -> tuple[float, float]:
    """How much of the input's power the speech stem kept, in dB, over the whole file and over the frames where the stem is active.

    Both are near 0 for a clean recording, and 0 when there is nothing to judge. The second is not fooled by
    long stretches of music between clean speech; the first is the one low enough to tell a recording with no voice.
    """
    ps, pm = _hop_powers(speech), _hop_powers(mix)
    if not len(pm) or pm.mean() <= 1e-13:
        return 0.0, 0.0
    # A clap or a cough in the recording says nothing about how much of the voice the stem kept.
    voice = gate.speech_level(mix)
    if voice is not None:
        normal = pm <= 10 ** ((voice + RECORDING_OVER_DB) / 10)
        ps, pm = ps[normal], pm[normal]
    active = ps > np.percentile(ps, 99.5) * 10 ** (-ACTIVE_DB / 10)
    whole = 10 * np.log10((ps.mean() + 1e-30) / pm.mean())
    during = 10 * np.log10((ps[active].sum() + 1e-30) / (pm[active].sum() + 1e-30)) if active.any() else whole
    return float(whole), float(during)


def stem_gain(stem: np.ndarray, whole_share_db: float = 0.0, target_lufs: float = TARGET_LUFS, skip: np.ndarray | None = None, lifted: float = 1.0) -> float:
    """Linear gain that brings the enhanced stem to the fixed target. Silence stays as it is.

    A recording with no voice is not lifted to speech level: when the speech stem is more than 25-35 dB
    below the mix over the whole file, it is separation residue. `skip` marks the 100 ms hops of loud
    moments, which would otherwise set the file's loudness and leave the speech far under the target.
    `lifted`: what a very quiet recording was multiplied by before the chain, which what stays as it is has to give back.
    """
    loudness = integrated_lufs(stem, skip)
    if not np.isfinite(loudness):
        return 1.0 / lifted
    return float(10 ** ((target_lufs - loudness + 20 * np.log10(lifted)) * np.clip((whole_share_db + 35) / 10, 0, 1) / 20) / lifted)

