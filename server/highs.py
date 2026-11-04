"""Fricative highs: gives regenerated speech the brightness of the recording's own fricatives.

Sidon hears only up to 8 kHz, so how bright its fricatives are above that is a guess. This step measures
the recording's own fricative level once per file and moves the restored stem towards it, in fricative
frames only. Vowels stay as they are.

Two functions, because a file is measured once and processed in blocks:

    plan = estimate_highs(restored, isolated, side=fused)     # once per file, reads in chunks, bounded memory
    y = apply_highs(restored[lo:hi], isolated[lo:hi], plan)   # per block, inside map_blocks

Where the recording cannot be trusted the plan leaves Sidon alone: above the recording's real bandwidth,
where hiss fills the band, with too little fricative evidence, and where different parts of the file
disagree. Energy that a codec has copied up (spectral band replication) shows as a rise in the vowels' highs
too, and then only the level below the hole is believed.
"""
from __future__ import annotations

import numpy as np
import librosa
from scipy.ndimage import gaussian_filter1d

SR = 48000
NF, HOP = 2048, 480
FQ = np.fft.rfftfreq(NF, 1 / SR)
EPS = 1e-14

# Bands that may move. Sidon's own level at 16 kHz and above matched the reference within 2.5 dB on both voices, so they stay.
CENTERS = np.array([4000.0, 5000.0, 6300.0, 8000.0, 10000.0, 12500.0])
BOTTOM, TOP = 3150.0, 16000.0
# Below 8 kHz the hybrid is partly the recording's own waveform and Sidon could hear the input, so the correction goes in part of the way.
SHARE = np.maximum(0.5, 1 - (0.5 + 0.5 * np.cos(np.pi * np.clip((CENTERS - 4000.0) / 4000.0, 0, 1))))
BOOST_DB, CUT_DB = 10.0, 6.0
# Asked to cut further than this, the recording is short of highs rather than Sidon too bright, so the cut fades back to nothing.
DEFICIT_DB = 24.0
# A voice's fricatives fall by at most about 10 dB from one third-octave band to the next on the voices measured. Steeper is a rolloff.
FALL_DB = 12.0
# A frame this far over the file's loud level is a loud moment, a word close to the mic for instance. It would outweigh all the speech in the power means.
LOUD_DB = 10.0
# A band needs this much more fricative power than noise power (6 dB) to be believed.
SNR_MIN = 4.0
FRICATIVE_FRAMES_NONE, FRICATIVE_FRAMES_FULL = 5, 40
GROUP_FRAMES, GROUPS_MIN, MIXED_FULL_DB, MIXED_NONE_DB = 40, 8, 8.0, 11.0
RISE_DB = 10.0

_edges = [(c / 2 ** (1 / 6), c * 2 ** (1 / 6)) for c in CENTERS]
_uppers = np.array([b for _, b in _edges])
_bands = [(FQ >= a) & (FQ < b) for a, b in _edges]
_hi, _lo, _mid = (FQ >= 3500) & (FQ < 12000), (FQ >= 200) & (FQ < 2000), (FQ >= 300) & (FQ < 3000)
# A 250 Hz grid for finding the recording's real bandwidth.
_grid = np.arange(2000.0, 23750.0, 250.0)
_fine = [(FQ >= g) & (FQ < g + 250) for g in _grid]


def _stft(x: np.ndarray) -> np.ndarray:
    return librosa.stft(np.asarray(x, dtype=np.float32), n_fft=NF, hop_length=HOP)


def _power(x: np.ndarray) -> np.ndarray:
    """Power spectrogram for measuring: whole frames only, so a chunk's cut edge cannot splatter energy across the highs."""
    return np.abs(librosa.stft(np.asarray(x, dtype=np.float32), n_fft=NF, hop_length=HOP, center=False)) ** 2


def _db(p: np.ndarray) -> np.ndarray:
    return 10 * np.log10(p + EPS)


def _spans(n: int, seconds_max: float, chunk_s: float = 30.0) -> list[tuple[int, int]]:
    """Chunks to read: the whole file when it is short, otherwise chunks spread evenly through it."""
    step, limit = int(chunk_s * SR), int(seconds_max * SR)
    if n <= limit:
        return [(a, min(n, a + step)) for a in range(0, n, step)]
    return [(int(a), int(a) + step) for a in np.linspace(0, n - step, max(1, limit // step))]


def _classify(total_db: np.ndarray, hi: np.ndarray, lo: np.ndarray, ref_db: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame fricative weight (0 to 1, smooth in time) and vowel flag, from the isolated stem's own spectrum."""
    balance = _db(hi) - _db(lo)
    fric = np.clip((balance + 3) / 6, 0, 1) * (total_db > ref_db - 30)
    vowel = (balance < -10) & (total_db > ref_db - 15)
    return gaussian_filter1d(fric.astype(np.float32), 1.5), vowel


def _cliff(fine_db: np.ndarray, cliff_db: float = 20.0) -> float:
    """Where a spectrum falls off a cliff and never comes back, in Hz, or the top of the grid.

    A voice falls by a few dB per kHz up here, a codec's lowpass by tens of dB. Requiring that nothing above
    returns to near the level just below keeps a hole with energy above it (spectral band replication) from
    reading as a cutoff.
    """
    tops = np.maximum.accumulate(fine_db[::-1])[::-1]
    for i in range(3, len(fine_db) - 1):
        if fine_db[i - 3 : i].mean() - tops[i + 1] >= cliff_db:
            return float(_grid[i])
    return float(_grid[-1] + 250.0)


def measure_bandwidth(isolated: np.ndarray, seconds_max: float = 600.0) -> float:
    """The recording's real bandwidth in Hz, from the long-term spectrum of its active frames."""
    total, count, parts = np.zeros(len(_fine)), 0, []
    for a, b in _spans(len(isolated), seconds_max):
        if b - a < NF:
            continue
        P = _power(isolated[a:b])
        parts.append((_db(P.sum(axis=0)), np.stack([P[m].mean(axis=0) for m in _fine]).astype(np.float32)))
    if not parts:
        return float(_grid[-1] + 250.0)
    levels = np.concatenate([p[0] for p in parts])
    ref = np.percentile(levels, 95)
    for lv, fine in parts:
        keep = lv > ref - 30
        total += fine[:, keep].sum(axis=1)
        count += int(keep.sum())
    return _cliff(_db(total / max(1, count)))


def _groups(on: np.ndarray, need: int) -> list[tuple[int, int]]:
    """Consecutive stretches of frames holding at least `need` fricative frames each."""
    out, start, seen = [], 0, 0
    for i in np.flatnonzero(on):
        seen += 1
        if seen == need:
            out.append((start, i + 1))
            start, seen = i + 1, 0
    return out


def identity_plan(reason: str, **extra) -> dict:
    return {"gains_db": np.zeros(len(CENTERS)), "ref_db": 0.0, "bandwidth_hz": 0.0, "fricative_frames": 0, "trust": 0.0, "reason": reason} | extra


def estimate_highs(restored: np.ndarray, isolated: np.ndarray, side: np.ndarray | None = None, seconds_max: float = 600.0, bandwidth: float | None = None) -> dict:
    """Measures the correction once per file. Accepts memmaps; reads at most seconds_max of audio.

    restored: the stem to correct. isolated: the recording's cleaned speech, the source of the target.
    side: the stem that shows the recording's noise (the speech before the second denoiser pass); defaults to isolated.
    bandwidth: from measure_bandwidth, if it has been measured already.
    """
    n = min(len(restored), len(isolated))
    if n < NF:
        return identity_plan("too short to measure")
    rows: dict[str, list] = {k: [] for k in ("total", "hi", "lo", "mx", "mi", "bx", "bi", "bs")}
    sizes = []
    for a, b in _spans(n, seconds_max):
        if b - a < NF:
            continue
        PX, PI = _power(restored[a:b]), _power(isolated[a:b])
        PS = PI if side is None else _power(side[a:b])
        k = PX.shape[1]
        rows["total"].append(_db(PI.sum(axis=0)))
        rows["hi"].append(PI[_hi].sum(axis=0))
        rows["lo"].append(PI[_lo].sum(axis=0))
        rows["mx"].append(PX[_mid].mean(axis=0))
        rows["mi"].append(PI[_mid].mean(axis=0))
        rows["bx"].append(np.stack([PX[m].mean(axis=0) for m in _bands]))
        rows["bi"].append(np.stack([PI[m].mean(axis=0) for m in _bands]))
        rows["bs"].append(np.stack([PS[m].mean(axis=0) for m in _bands]))
        sizes.append(k)
    total = np.concatenate(rows["total"])
    ref_db = float(np.percentile(total, 95))
    if ref_db < -40:
        return identity_plan("silence")
    hi, lo = np.concatenate(rows["hi"]), np.concatenate(rows["lo"])
    fric, vowel, at = [], [], 0
    for k in sizes:
        f, v = _classify(total[at : at + k], hi[at : at + k], lo[at : at + k], ref_db)
        fric.append(f)
        vowel.append(v)
        at += k
    speech = total <= ref_db + LOUD_DB
    fric, vowel = np.concatenate(fric), np.concatenate(vowel) & speech
    on = (fric > 0.6) & speech
    count = int(on.sum())
    if bandwidth is None:
        bandwidth = measure_bandwidth(isolated, seconds_max)
    plan = identity_plan("too few fricatives or vowels", ref_db=ref_db, fricative_frames=count, bandwidth_hz=bandwidth)
    if count <= FRICATIVE_FRAMES_NONE or vowel.sum() < 10:
        return plan
    mx, mi = np.concatenate(rows["mx"]), np.concatenate(rows["mi"])
    bx, bi, bs = (np.concatenate(rows[k], axis=1) for k in ("bx", "bi", "bs"))
    # Vowels and pauses are quiet this high up, so a low percentile finds the noise even in a file with no gaps.
    floor = 10 ** (np.percentile(_db(bs), 15, axis=1) / 10)
    # An /s/ can rise with frequency, but a vowel's highs fall. Energy that rises in vowels too has been copied up by a codec, and only
    # the level below the hole can be believed.
    vowel_levels = np.median(_db(bi[:, vowel]), axis=1)
    replicated = bool((vowel_levels - np.minimum.accumulate(vowel_levels)).max() > RISE_DB)

    def correction(frames: np.ndarray, vowels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Target minus restored per band, and whether each band can be believed. Hiss reads as brightness, so the floor comes off."""
        offset = _db(mx[vowels].mean()) - _db(mi[vowels].mean())
        fx, fi = bx[:, frames].mean(axis=1), bi[:, frames].mean(axis=1)
        clean = np.maximum(fi - floor, 0)
        ok = (_uppers * 0.95 <= bandwidth) & (clean > floor * SNR_MIN)
        levels = _db(clean)
        # From 5 kHz up: an /s/ can start high enough that 4 to 5 kHz falls steeply, and a real rolloff there fails the two tests above.
        ok[2:] &= levels[2:] - levels[1:-1] > -FALL_DB
        ok = np.logical_and.accumulate(ok)  # Everything above the first band without real signal stays with Sidon.
        return (np.minimum.accumulate(levels) if replicated else levels) + offset - _db(fx), ok

    raw, ok = correction(on, vowel)
    spread = 0.0
    groups = _groups(on, GROUP_FRAMES)
    if len(groups) >= GROUPS_MIN:
        each = []
        for a, b in groups:
            span = np.zeros(len(on), bool)
            span[a:b] = True
            each.append(correction(on & span, (vowel & span) if (vowel & span).sum() >= 5 else vowel)[0])
        p20, p80 = np.percentile(each, [20, 80], axis=0)
        spread = float((p80 - p20)[2:5].mean())
    trust = float(np.clip((count - FRICATIVE_FRAMES_NONE) / (FRICATIVE_FRAMES_FULL - FRICATIVE_FRAMES_NONE), 0, 1))
    # Parts of the file that want different corrections are different sources (a remote guest, a codec in part of it). One correction cannot suit both.
    trust *= float(np.clip((MIXED_NONE_DB - spread) / (MIXED_NONE_DB - MIXED_FULL_DB), 0, 1))
    fade = np.clip((DEFICIT_DB + raw) / (DEFICIT_DB - CUT_DB), 0, 1)
    gains = np.clip(raw, -CUT_DB, BOOST_DB) * fade * SHARE * trust * ok
    plan.update(gains_db=gains, trust=trust, reason="measured", raw_db=raw, usable=ok, spread_db=spread, replicated=replicated, snr_db=_db(bi[:, on].mean(axis=1)) - _db(floor))
    return plan


def _bin_gains(gains_db: np.ndarray) -> np.ndarray:
    """dB gain per STFT bin: the band gains joined up on a log frequency axis, 0 at BOTTOM and from TOP upward."""
    return np.interp(np.log2(np.maximum(FQ, 1.0)), np.log2(np.r_[BOTTOM, CENTERS, TOP]), np.r_[0.0, gains_db, 0.0], left=0.0, right=0.0)


def apply_highs(restored: np.ndarray, isolated: np.ndarray, plan: dict) -> np.ndarray:
    """Applies a plan to one block. Matches a whole-file run when the block starts on the 480-sample grid and carries 250 ms of margin."""
    if not np.any(plan["gains_db"]):
        return np.asarray(restored, dtype=np.float32)
    X, I = _stft(restored), _stft(isolated)
    k = min(X.shape[1], I.shape[1])
    PI = np.abs(I[:, :k]) ** 2
    fric, _ = _classify(_db(PI.sum(axis=0)), PI[_hi].sum(axis=0), PI[_lo].sum(axis=0), plan["ref_db"])
    gdb = _bin_gains(plan["gains_db"])
    on = np.flatnonzero(gdb != 0)
    X[on[0] : on[-1] + 1, :k] *= 10 ** (np.outer(gdb[on[0] : on[-1] + 1], fric) / 20)
    return librosa.istft(X, hop_length=HOP, n_fft=NF, length=len(restored)).astype(np.float32)
