"""The enhancement chain. Stems live in memory-mapped raw files, so long recordings stay within RAM.

v2 (default): BandIt v2 + DeepFilterNet3 -> Sidon -> waveform and envelope blend -> harmonic clarity -> tone -> highs -> limit by the recording -> gates -> loudness -> floor
v1 (quick):   DeepFilterNet3 -> MossFormer2 -> rumble filter -> gate -> loudness, blended with the original

bench/ holds the scripts that chose these steps by scoring them against reference enhanced tracks.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from . import clarity, dsp, gate, highs, level, limit, media, tone
from .engines import SR, get_engines, speech_regions

Report = Callable[[str, float], None]

# Models and measures in the chain have fixed floors: DeepFilterNet removes speech under about -40 dBFS as if it were silence. So a recording whose
# median voiced frame is under QUIET_DB runs through at QUIET_TO_DB, and its stems come back down at the end. The quietest of ten real recordings sits at -27 dB.
QUIET_DB, QUIET_TO_DB = -30.0, -20.0
# Bytes a job holds on disk at its peak, per sample of audio: its float32 stems and the merged file. About 21 and 12 GB per hour.
ROOM_PER_SAMPLE = {"v2": 4 * 31, "v1": 4 * 18}


def _check_room(work: Path, samples: float, mode: str, written: int = 0) -> None:
    """Stems are memory-mapped, so a full disk would only show when they are written, hours into a long file. `written`: bytes of the job's already on disk."""
    need, free = samples * ROOM_PER_SAMPLE[mode], shutil.disk_usage(work).free + written
    if free < need:
        raise media.MediaError(f"This file needs about {need / 1e9:.1f} GB of free disk space while it processes, and {free / 1e9:.1f} GB is free. Free up some space and try again.")


def _open(path: Path, n: int, channels: int, mode: str = "w+") -> np.memmap:
    shape = (n, channels) if channels > 1 else (n,)
    return np.memmap(path, dtype="float32", mode=mode, shape=shape)


def map_blocks(fn, n: int, outputs: Sequence[np.ndarray], block: int, margin: int, fade: int, progress: Callable[[float], None]) -> None:
    """Runs fn(lo, hi, progress) over overlapping spans of n samples and cross-fades the results into outputs.

    Each span carries `margin` samples of extra context on both sides, so models never start cold at
    a seam. fn returns one array per output, each covering exactly [lo, hi).
    """
    starts = [0] if n <= block + margin else list(range(0, n, block))
    if len(starts) > 1 and n - starts[-1] < margin:
        starts.pop()
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    for i, start in enumerate(starts):
        end = n if i == len(starts) - 1 else start + block
        lo, hi = max(0, start - margin), min(n, end + margin)
        results = fn(lo, hi, lambda f, i=i: progress((i + f) / len(starts)))
        results = results if isinstance(results, tuple) else (results,)
        for y, dst in zip(results, outputs):
            core = y[start - lo : end - lo]
            if start > 0:
                r = ramp if core.ndim == 1 else ramp[:, None]
                core = core.copy()
                core[:fade] = dst[start : start + fade] * (1 - r) + core[:fade] * r
            dst[start:end] = core
            # The next span fades in over these extra samples.
            tail = min(n, end + fade)
            if tail > end:
                dst[end:tail] = y[end - lo : tail - lo]
        progress((i + 1) / len(starts))
    for dst in outputs:
        dst.flush()


def _channel_choice(stereo: np.memmap) -> int | None:
    """The channel that stands for the recording, or None to average the two. Decided once for the whole file, so no block differs from the next.

    A dead channel would otherwise cost 6 dB, and a channel wired out of phase would cancel the voice. A channel
    counts as dead only if it is dead in every minute, so a guest who says little on their own mic is kept.
    """
    left = right = cross = 0.0
    left_dead = right_dead = True
    for start in range(0, len(stereo), SR * 60):
        block = np.asarray(stereo[start : start + SR * 60], dtype=np.float64)
        el, er = float((block[:, 0] ** 2).sum()), float((block[:, 1] ** 2).sum())
        left_dead &= el <= er * 1e-4
        right_dead &= er <= el * 1e-4
        left, right = left + el, right + er
        cross += float((block[:, 0] * block[:, 1]).sum())
    if left_dead != right_dead:
        return 1 if left_dead else 0
    if cross < -0.5 * np.sqrt(left * right):
        return 0 if left >= right else 1
    return None


def _downmix(stereo: np.ndarray, channel: int | None) -> np.ndarray:
    return ((stereo[:, 0] + stereo[:, 1]) / 2).astype(np.float32) if channel is None else stereo[:, channel].copy()


def _filtered(work: Path, stem: np.memmap, taps: np.ndarray, n: int, report: Report) -> np.memmap:
    out = _open(work / "toned.raw", n, 1)
    map_blocks(
        lambda lo, hi, _p: dsp.apply_fir(np.asarray(stem[lo:hi]), taps),
        n, [out], SR * 120, len(taps), SR // 100, lambda f: report("finishing", 0.15 * f),
    )
    return out


def _finish_enhanced(work: Path, stem: np.memmap, whole: float, share: float, n: int, report: Report, lift: float, v2: bool = False) -> np.memmap:
    """Closes the gaps between words, then sets the stem's loudness and limits its peaks.

    `whole` is the speech's share of the input over the file, `share` the same while someone speaks, where the separator lost the voice.
    v2 also gets a gate for the low band and a noise floor. Both were measured on v2 reference output, and no v1 reference exists.
    """
    reference, strength = gate.speech_reference(stem), gate.strength_for(share)
    low, low_strength = (gate.low_reference(stem, reference[2]), strength * limit.strength_for(whole)) if v2 else (-140.0, 0.0)

    def close(lo: int, hi: int, _p) -> np.ndarray:
        y = gate.speech_gate(np.asarray(stem[lo:hi]), reference, start=lo, strength=strength)
        return gate.low_gate(y, low, reference, start=lo, strength=low_strength)

    gated = _open(work / "gated.raw", n, 1)
    map_blocks(close, n, [gated], SR * 60, SR * 2, SR // 100, lambda f: report("finishing", 0.3 + 0.2 * f))
    gain = level.stem_gain(gated, whole, skip=gate.loud_hops(reference[2]), lifted=lift)

    def finish(lo: int, hi: int, _p) -> np.ndarray:
        y = np.asarray(gated[lo:hi]) * gain * gate.loud_gain(reference[2], lo, hi - lo)
        return dsp.limiter(y + gate.gap_floor(lo, hi - lo, reference[0], gain) if v2 else y)

    final = _open(work / "enhanced.raw", n, 1)
    map_blocks(finish, n, [final], SR * 120, SR // 10, SR // 100, lambda f: report("finishing", 0.5 + 0.1 * f))
    return final


def _run_v2(work: Path, original: np.memmap, mono: np.memmap, channel: int | None, n: int, report: Report, lift: float):
    engines = get_engines()
    isolated = _open(work / "isolated.raw", n, 1)
    music = _open(work / "music.raw", n, 2)
    background = _open(work / "background.raw", n, 2)
    fused = _open(work / "fused.raw", n, 1)
    from_fallback, voice = 0.0, 0.0

    def separate(lo: int, hi: int, progress) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nonlocal from_fallback, voice
        stems = engines.separator(np.asarray(original[lo:hi]), progress)
        split = _downmix(stems["speech"], channel)
        # Where the separator lost the voice, the denoiser has it. Under heavy noise and echo that
        # can be the whole recording.
        safe = engines.fast_denoiser(np.asarray(mono[lo:hi]))
        weight = dsp.fallback_weight(split, safe, speech_regions(safe))
        mixed = (1 - weight) * split + weight * safe
        energy = safe.astype(np.float64) ** 2
        from_fallback, voice = from_fallback + float(np.dot(weight, energy)), voice + float(energy.sum())
        # The separator can also leave room noise in the speech, which would be audible at the
        # slider's middle positions.
        speech = engines.fast_denoiser(mixed)
        # Whatever left the speech goes to the background, so the stems still add up to the original.
        return speech, stems["music"], stems["sfx"] + (split - speech)[:, None], mixed

    map_blocks(separate, n, [isolated, music, background, fused], SR * 300, SR * 4, SR // 10, lambda f: report("separating_background", f))

    # Sidon hears the speech before the second denoiser pass, which blurs weak syllables into each other.
    restored = _open(work / "restored.raw", n, 1)
    map_blocks(
        lambda lo, hi, p: level.follow_input(engines.restorer(np.asarray(fused[lo:hi]), p), np.asarray(isolated[lo:hi])),
        n, [restored], SR * 30, SR * 2, SR // 20, lambda f: report("enhancing", f),
    )

    shaped = _open(work / "shaped.raw", n, 1)
    map_blocks(
        lambda lo, hi, _p: dsp.hybrid(np.asarray(isolated[lo:hi]), np.asarray(restored[lo:hi]), np.asarray(mono[lo:hi])),
        n, [shaped], SR * 60, SR * 2, SR // 50, lambda f: report("removing_echo", 0.5 * f),
    )

    whole, during = level.speech_share(isolated, mono)
    plan = clarity.estimate_clarity(shaped, isolated, during)
    clear = _open(work / "clear.raw", n, 1)
    map_blocks(
        lambda lo, hi, _p: clarity.apply_clarity(np.asarray(shaped[lo:hi]), np.asarray(isolated[lo:hi]), plan),
        n, [clear], SR * 60, SR * 2, SR // 50, lambda f: report("removing_echo", 0.5 + 0.5 * f),
    )

    taps, _ = tone.tone_fir(clear, isolated)
    toned = _filtered(work, clear, taps, n, report)
    # Measured once on the whole file. The blocks stay on the 480-sample grid its frames use.
    plan_highs = highs.estimate_highs(toned, isolated, side=fused)
    bright = _open(work / "bright.raw", n, 1)
    map_blocks(
        lambda lo, hi, _p: highs.apply_highs(np.asarray(toned[lo:hi]), np.asarray(isolated[lo:hi]), plan_highs),
        n, [bright], SR * 60, SR // 2, SR // 100, lambda f: report("finishing", 0.15 + 0.1 * f),
    )
    plan_limit = limit.limit_plan(bright, mono, limit.strength_for(whole))
    limited = _open(work / "limited.raw", n, 1)
    map_blocks(
        lambda lo, hi, _p: limit.limit_block(np.asarray(bright[lo:hi]), np.asarray(mono[lo:hi]), plan_limit),
        n, [limited], SR * 60, SR // 2, SR // 100, lambda f: report("finishing", 0.25 + 0.05 * f),
    )
    # A low share only counts as noise if the separator could not find the voice: it handles loud music well, and files babble and distortion under "music".
    share = during * from_fallback / max(voice, 1e-12)
    return _finish_enhanced(work, limited, whole, share, n, report, lift, v2=True), isolated, music, background, share


def _run_v1(work: Path, original: np.memmap, mono: np.memmap, channel: int | None, n: int, report: Report, lift: float):
    engines = get_engines()
    fast = _open(work / "fast.raw", n, 1)
    map_blocks(
        lambda lo, hi, _p: engines.fast_denoiser(np.asarray(mono[lo:hi])),
        n, [fast], SR * 300, SR, SR // 20, lambda f: report("removing_echo", f),
    )
    cleaned = _open(work / "cleaned.raw", n, 1)
    map_blocks(
        lambda lo, hi, _p: engines.denoiser(np.asarray(fast[lo:hi])),
        n, [cleaned], SR * 120, SR * 2, SR // 20, lambda f: report("enhancing", f),
    )
    toned = _filtered(work, cleaned, tone.rumble_fir(), n, report)
    # v1 has no separator to vouch for the voice, so a low share always counts.
    whole, share = level.speech_share(cleaned, mono)
    return _finish_enhanced(work, toned, whole, share, n, report, lift), None, None, None, share


def process(source: Path, work: Path, out: Path, mode: str, report: Report) -> dict:
    """Runs the full chain, writes the merged preview file to `out`, and returns track metadata.

    `report(stage, fraction)` is called throughout and may raise to cancel the job.
    """
    info = media.probe(source)
    if not info.has_audio:
        raise media.MediaError("The file contains no audio to enhance.")

    # The track's folder exists until someone deletes it; a parent made here would outlive that.
    work.mkdir(exist_ok=True)
    report("processing", 0.0)
    _check_room(work, info.duration * SR, mode)
    media.decode_stereo(source, work / "original.raw", info.audio_channels)
    n = (work / "original.raw").stat().st_size // 8
    if n == 0:
        raise media.MediaError("The file contains no audio to enhance.")
    # Some files do not say how long they are until they are decoded.
    _check_room(work, n, mode, written=n * 8)
    original = _open(work / "original.raw", n, 2, "r")
    channel = _channel_choice(original)
    mono = _open(work / "mono.raw", n, 1)
    for start in range(0, n, SR * 60):
        mono[start : start + SR * 60] = _downmix(np.asarray(original[start : start + SR * 60]), channel)
    voice = gate.speech_level(mono)
    lift, stereo = 1.0, original
    if voice is not None and voice < QUIET_DB:
        lift = 10 ** ((QUIET_TO_DB - voice) / 20)
        stereo = _open(work / "lifted.raw", n, 2)
        for start in range(0, n, SR * 60):
            stereo[start : start + SR * 60] = np.asarray(original[start : start + SR * 60]) * lift
            mono[start : start + SR * 60] *= lift
        stereo.flush()
    mono.flush()
    report("processing", 1.0)

    run = _run_v2 if mode == "v2" else _run_v1
    enhanced, isolated, music, background, share = run(work, stereo, mono, channel, n, report, lift)

    # Channel order the player expects: original L R, enhanced, isolated, music L R, background L R.
    merged_raw = work / "merged.raw"
    with open(merged_raw, "wb") as f:
        step = SR * 30
        for start in range(0, n, step):
            end = min(n, start + step)
            frame = np.zeros((end - start, 8), dtype=np.float32)
            frame[:, 0:2] = original[start:end]
            frame[:, 2] = enhanced[start:end]
            if isolated is not None:
                frame[:, 3] = isolated[start:end] / lift
                frame[:, 4:6] = music[start:end] / lift
                frame[:, 6:8] = background[start:end] / lift
            f.write(frame.tobytes())
            report("finishing", 0.6 + 0.2 * end / n)
    media.build_merged(merged_raw, source, info, out)
    report("finishing", 1.0)
    return {"duration": n / SR, "isVideo": info.has_video, "speechShare": share, "rawChannel": channel}
