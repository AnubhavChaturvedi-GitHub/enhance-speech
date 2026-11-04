"""Speech gate: black gaps between words without clipping quiet speech. Its rules were measured on a reference enhanced track, on two voices.

The gain depends only on the local signal (within about 1.2 s) and on the file's speech reference, so the output does not depend on
where a block starts as long as the margin is at least 2 s.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.signal import butter, fftconvolve, firwin, sosfiltfilt

SR = 48000
_F = 48  # gain frame: 1 ms
_REF_HOP = SR // 5  # the reference track has one value per 200 ms

# Every level below is in dB relative to the speech reference (the loud frames of the talker).
CLOSE = -38.0        # the expander starts here
SLOPE = 2.5          # dB of gain per dB below the threshold
DEPTH = 40.0         # most attenuation; a breath at -30 lands at -70, about the reference's floor above 80 Hz
EDGE = -30.0         # a fall or rise that passes from here to FLOOR in under EDGE_MS is expanded against this level
EDGE_MS = 20
FLOOR = -42.0        # a run of sound is what stays above this level
SPEECH = -20.0       # a run that reaches this level contains speech and is never removed
SEED = -32.0         # a run under BLIP_MS that never reaches this level is a blip
BLIP_MS = 50
BREATH_GAP_MS = 120  # a breath sits this far from speech on at least one side...
BREATH_RIGHT_MS = 30 # ...and does not run straight into the next word
BREATH_MIN = -37.0   # quieter runs are left to the expander
BREATH_MAX_MS = 1000 # longer runs are not breaths
UP = 10.0            # dB/ms: how fast the gain may open
DOWN = 2.0           # dB/ms: how fast the gain may close
NOISY_DB = (-10.0, -6.0)  # speech stem against the input: no gate below the first, the full gate above the second
# A fricative or a gap holds no low band, but the regenerated stem leaves some there, so the band under LOW_HZ gets its own expander against its own loud level.
LOW_HZ = 500.0
LOW_THRESHOLD = -25.0  # relative to the band's loud level (low_reference), lowered where speech_reference's track is
LOW_DEPTH = 20.0
LOW_UP, LOW_DOWN = 3.0, 1.0  # dB/ms
_LOW_TAPS = firwin(1441, LOW_HZ, fs=SR).astype(np.float32)
# The reference output's gaps are not black: all three voices hold a nearly white floor this far under the speech level.
FLOOR_DB = -72.0
_FLOOR_SAMPLES = 1 << 19  # about 11 s, repeats by position
_floor: np.ndarray | None = None
# 100 ms this far over the stem's median voiced frame is a loud moment, a cough or a word close to the mic, not the speech. Enhanced speech reaches 7-9 dB on three voices.
LOUD_OVER = 10.0
LOUD_TO = 12.0  # what a loud moment holds above this comes down to it, about the talker's own loudest frames
_MID = butter(4, [1000, 4000], btype="band", fs=SR, output="sos")
_HIGH = butter(4, [4000, 12000], btype="band", fs=SR, output="sos")
_LOW = butter(4, 1500, btype="low", fs=SR, output="sos")


def _levels(x: np.ndarray, window: int, hop: int) -> np.ndarray:
    # The running sum can leave a tiny negative rounding residue after loud sound stops.
    return 10 * np.log10(np.maximum(uniform_filter1d(np.nan_to_num(np.asarray(x, dtype=np.float64)) ** 2, window), 0.0)[::hop] + 1e-14)


def _loud_level(frames_db: np.ndarray) -> float:
    """The 95th percentile of a file's frame levels, or the upper quartile of what is near its peak when sound fills under about 15% of the file."""
    return max(float(np.percentile(frames_db, 95)), float(np.percentile(frames_db[frames_db > np.percentile(frames_db, 99.5) - 30], 75)))


def _periodicity(x: np.ndarray, hop: int, window: int = 1920) -> np.ndarray:
    """Peak normalised autocorrelation per frame on a 1.5 kHz low-pass, 70-400 Hz lags. Near 0 for noise, near 1 for a held voice."""
    if len(x) < 64:
        return np.zeros(1)
    y = sosfiltfilt(_LOW, np.nan_to_num(x), padlen=min(27, len(x) - 1))[::6]
    h, size = hop // 6, window // 6
    pad = np.pad(y, (size // 2, size))
    n = len(y) // h + 1
    frames = np.stack([pad[i * h : i * h + size] for i in range(n)]).astype(np.float64)
    frames -= frames.mean(axis=1, keepdims=True)
    w = np.hanning(size)
    ac = np.fft.irfft(np.abs(np.fft.rfft(frames * w, 2 * size, axis=1)) ** 2, axis=1)[:, :size]
    acw = np.fft.irfft(np.abs(np.fft.rfft(w, 2 * size)) ** 2)[:size]
    r = ac / (ac[:, :1] + 1e-14) / (acw / acw[0] + 1e-6)
    lo, hi = int(8000 / 400), int(8000 / 70)
    return np.clip(r[:, lo:hi].max(axis=1), 0, 1) * (ac[:, 0] > 1e-13)


def _frames(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Level and periodicity per 10 ms of the file. Runs in chunks; x may be a memmap."""
    levels, voiced = [], []
    chunk = SR * 60
    for start in range(0, max(len(x), 1), chunk):
        block = np.asarray(x[max(0, start - 960) : start + chunk], dtype=np.float32)
        skip = 960 // 480 if start > 0 else 0
        levels.append(_levels(block, 960, 480)[skip:])
        # One frame shorter than _periodicity gives, so the two stay in step from chunk to chunk.
        voiced.append(_periodicity(block, 480)[skip:][: len(levels[-1])])
    lev, per = np.concatenate(levels), np.concatenate(voiced)
    k = min(len(lev), len(per))
    return lev[:k], per[:k]


def _voice_level(lev: np.ndarray, per: np.ndarray, reach: float = 30.0) -> float | None:
    """The median voiced frame's level, or None under half a second of voice. A loud moment hardly moves the median."""
    if not len(lev):
        return None
    voiced = lev[(per >= 0.6) & (lev > np.percentile(lev, 95) - reach)]
    return float(np.median(voiced)) if len(voiced) >= 50 else None


def speech_level(x: np.ndarray) -> float | None:
    """Where the file's speech sits in dB, whatever loud moments it holds. None if it holds too little voice to tell."""
    return _voice_level(*_frames(x))


def _loud_moments(lev: np.ndarray, voice: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Which 10 ms frames belong to a loud moment, and the cut in dB that brings them down. All clear on a file without one."""
    clear = np.zeros(len(lev), bool), np.zeros(len(lev))
    if voice is None:
        return clear
    loud = 10 * np.log10(np.maximum(uniform_filter1d(10 ** (lev / 10), 10), 1e-14)) > voice + LOUD_OVER
    # 100 ms each side, so the dips between a near talker's syllables do not stay louder than the syllables.
    loud = maximum_filter1d(loud, 21)
    if not loud.any() or loud.all():
        return clear
    cut = minimum_filter1d(np.where(loud, np.minimum(voice + LOUD_TO - lev, 0.0), 0.0), 3)
    k = np.hanning(7)[1:-1]
    return loud, np.convolve(cut, k / k.sum(), mode="same")


def speech_reference(x: np.ndarray, slack: float = 6.0, reach: float = 30.0, radius_s: float = 1.0) -> tuple[float, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Measures where speech sits: the file's level (20 ms frames, usually their 95th percentile), a slower local track, one value per 200 ms, and its loud moments.

    The track only departs from the file's level where the voiced speech nearby is more than `slack` dB quieter, for example a
    quiet second talker. Without this a global threshold would treat that talker's words as breaths. Loud moments count towards
    none of it. Runs in chunks; x may be a memmap.
    """
    n = len(x)
    lev, per = _frames(x)
    k = len(lev)
    moments = _loud_moments(lev, _voice_level(lev, per, reach))
    speech = lev[~moments[0]]
    typical = float(np.percentile(speech, 95)) if k else -140.0
    overall = _loud_level(speech) if k else typical
    steps = n // _REF_HOP + 2
    track = np.full(steps, overall, dtype=np.float32)
    # Voicing is judged from the typical level, so a talker far below one short loud event still gets a track of their own.
    usable = (per >= 0.6) & (lev > typical - reach) & ~moments[0]
    radius = int(radius_s * 100)
    per_step = _REF_HOP // 480
    for i in range(steps):
        c = i * per_step
        v = lev[max(0, c - radius) : c + radius][usable[max(0, c - radius) : c + radius]]
        if len(v) >= 20:
            track[i] = min(overall, float(np.percentile(v, 95)) + slack)
    return overall, track, moments


def loud_hops(moments: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Which 100 ms hops, as level.py counts them, touch a loud moment."""
    loud = moments[0]
    return loud[: len(loud) // 10 * 10].reshape(-1, 10).any(axis=1)


def loud_gain(moments: tuple[np.ndarray, np.ndarray], start: int, n: int) -> np.ndarray | float:
    """The gain that brings a file's loud moments down, for sample `start` onwards. Measured once per file, so the same whatever the block split."""
    cut = moments[1]
    if not cut.any():
        return 1.0
    return (10 ** (np.interp((start + np.arange(n)) / 480, np.arange(len(cut)), cut) / 20)).astype(np.float32)


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _is_breath(x: np.ndarray, a: int, b: int) -> bool:
    """A breath has more energy at 1-4 kHz than at 4-12 kHz (unlike a soft fricative) and no held voice (unlike a quiet word)."""
    seg = x[max(0, a * _F - 480) : b * _F + 480]
    if float((sosfiltfilt(_MID, seg, padlen=480) ** 2).sum()) <= float((sosfiltfilt(_HIGH, seg, padlen=480) ** 2).sum()):
        return False
    voiced_ms = 5 * int((_periodicity(x[a * _F : b * _F], 240) >= 0.6).sum())
    return voiced_ms < 30


def _gain(x: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Gain in dB per 1 ms frame, from the level relative to speech."""
    expand = lambda threshold: -np.clip((threshold - L) * SLOPE, 0, DEPTH)
    g = expand(CLOSE)
    # A short stay between FLOOR and EDGE is the voice falling or rising through; a long one is a sustained quiet sound and stays.
    a, b = _runs((L > FLOOR) & (L < EDGE))
    transit = np.zeros(len(L), bool)
    for s, e in zip(a[b - a < EDGE_MS], b[b - a < EDGE_MS]):
        transit[s:e] = True
    g = np.where(transit, expand(EDGE), g)
    # Close dips under 5 ms so a run is not broken by the level ripple.
    above = L > FLOOR
    if above.any():
        a, b = _runs(~above)
        for s, e in zip(a, b):
            if e - s < 5 and s > 0 and e < len(L):
                above[s:e] = True
    speech = np.flatnonzero(L >= SPEECH)
    for s, e in zip(*_runs(above)):
        peak = float(L[s:e].max())
        if peak >= SPEECH:
            continue
        length = e - s
        if length < BLIP_MS:
            if peak < SEED:
                g[s:e] = -DEPTH
            continue
        if length > BREATH_MAX_MS:
            continue
        j = int(np.searchsorted(speech, s))
        left = s - speech[j - 1] if j > 0 else 10**6
        right = speech[j] - e + 1 if j < len(speech) else 10**6
        mean = 10 * np.log10(np.mean(10 ** (L[s:e] / 10)))
        if max(left, right) >= BREATH_GAP_MS and right >= BREATH_RIGHT_MS and mean >= BREATH_MIN and _is_breath(x, s, e):
            g[s:e] = -DEPTH
    return g


def _slew(g: np.ndarray) -> np.ndarray:
    """Limits the gain's speed in dB: it closes at DOWN dB/ms after a word and opens at UP dB/ms before one, never later than needed."""
    y = g.tolist()
    for i in range(1, len(y)):
        v = y[i - 1] - DOWN
        if v > y[i]:
            y[i] = v
    for i in range(len(y) - 2, -1, -1):
        v = y[i + 1] - UP
        if v > y[i]:
            y[i] = v
    return np.asarray(y)


def strength_for(share_db: float) -> float:
    """How hard to gate, 0 to 1, from how much of the input the speech stem kept while someone spoke, where the separator lost the voice.

    Under noise this heavy the regenerated speech has holes, and what is quiet in it is often the words themselves.
    """
    return float(np.clip((share_db - NOISY_DB[0]) / (NOISY_DB[1] - NOISY_DB[0]), 0, 1))


def speech_gate(x: np.ndarray, reference: tuple | None = None, start: int = 0, strength: float = 1.0) -> np.ndarray:
    """Returns x with its gaps pushed to near silence. `reference` comes from speech_reference on the whole file; `start` is x's first sample in that file."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float32))
    if len(x) < 480 or strength <= 0:
        return x
    overall, track, _ = speech_reference(x) if reference is None else reference
    if overall < -80:  # silence, or too little sound to judge
        return x
    # Frames sit on whole milliseconds of the file, so any block start gives the same frames.
    offset = -start % _F
    frames = _levels(x, 480, 1)[offset::_F]
    ref = np.interp((start + offset + np.arange(len(frames)) * _F) / _REF_HOP, np.arange(len(track)), track)
    L = frames - ref
    g = _slew(_gain(x[offset:], L))
    k = np.hanning(7)[1:-1]  # 2 ms each side: no click
    g = np.convolve(np.pad(g, 2, mode="edge"), k / k.sum(), mode="valid")
    gain = 10 ** (strength * np.interp((np.arange(len(x)) - offset) / _F, np.arange(len(g)), g) / 20)
    return (x * gain).astype(np.float32)


def _low_band(x: np.ndarray) -> np.ndarray:
    return fftconvolve(x, _LOW_TAPS, mode="same").astype(np.float32)


def low_reference(stem: np.ndarray, moments: tuple[np.ndarray, np.ndarray]) -> float:
    """The low band's loud level in dB, once per file, leaving out loud moments. Reads in chunks; stem may be a memmap."""
    levels = []
    for start in range(0, max(len(stem), 1), SR * 60):
        block = np.nan_to_num(np.asarray(stem[start : start + SR * 60], dtype=np.float32))
        if len(block) >= 960:
            levels.append(_levels(_low_band(block), 960, 480))
    if not levels:
        return -140.0
    low = np.concatenate(levels)
    keep = np.ones(len(low), bool)
    keep[: len(moments[0])] = ~moments[0][: len(low)]
    return _loud_level(low[keep])


def low_gate(x: np.ndarray, reference_db: float, speech: tuple, start: int = 0, strength: float = 1.0) -> np.ndarray:
    """Expands the band under LOW_HZ against its own loud level, lowered as far as speech_reference's track says the talker nearby is quieter than the file.

    The same whatever the block split, with at least 100 ms of margin.
    """
    x = np.nan_to_num(np.asarray(x, dtype=np.float32))
    if len(x) < 480 or strength <= 0 or reference_db < -80:
        return x
    low = _low_band(x)
    offset = -start % _F
    frames = _levels(low, 960, 1)[offset::_F]
    overall, track, _ = speech
    ref = reference_db + np.interp((start + offset + np.arange(len(frames)) * _F) / _REF_HOP, np.arange(len(track)), track) - overall
    g = (-np.clip(LOW_THRESHOLD - (frames - ref), 0, LOW_DEPTH)).tolist()
    for i in range(1, len(g)):
        g[i] = max(g[i], g[i - 1] - LOW_DOWN)
    for i in range(len(g) - 2, -1, -1):
        g[i] = max(g[i], g[i + 1] - LOW_UP)
    k = np.hanning(7)[1:-1]
    g = np.convolve(np.pad(np.asarray(g), 2, mode="edge"), k / k.sum(), mode="valid")
    gain = 10 ** (strength * np.interp((np.arange(len(x)) - offset) / _F, np.arange(len(g)), g) / 20)
    return (low * gain.astype(np.float32) + (x - low)).astype(np.float32)


def gap_floor(start: int, n: int, overall_db: float, gain: float = 1.0) -> np.ndarray:
    """A noise floor for sample `start` onwards, the same whatever the block split. overall_db: speech_reference's level. gain: the stem's loudness gain. Silence gets none."""
    global _floor
    if overall_db < -80:
        return np.zeros(n, np.float32)
    if _floor is None:
        _floor = np.random.default_rng(20261103).standard_normal(_FLOOR_SAMPLES).astype(np.float32)
    return _floor[(start + np.arange(n)) % _FLOOR_SAMPLES] * float(10 ** ((overall_db + FLOOR_DB) / 20) * gain)
