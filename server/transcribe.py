"""Local transcripts and captions with faster-whisper, read from the enhanced speech unless noise buried the voice."""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from . import engines, media

MODEL = "small"
# Below this share of the input (see pipeline._finish_enhanced), the regenerated speech may have lost words the raw recording still has.
SUSPECT_BELOW_DB = -8.0
_lock = threading.Lock()
_model = None


def _timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def _read(folder: Path, channels: str) -> list:
    """Whisper's segments for a mono mix of the merged file's channels."""
    raw = folder / "speech16k.raw"
    try:
        media.decode_for_speech(folder / "merged.mp4", channels, raw)
        segments, _ = _model.transcribe(np.fromfile(raw, dtype="float32"), beam_size=5, vad_filter=True)
        return list(segments)
    finally:
        raw.unlink(missing_ok=True)


def _confidence(segments: list) -> float:
    """Whisper's mean log probability, weighted by the length of each segment's text."""
    length = sum(len(s.text) for s in segments)
    return sum(s.avg_logprob * len(s.text) for s in segments) / length if length else float("-inf")


def transcript_files(folder: Path, speech_share: float = 0.0, channel: int | None = None) -> tuple[Path, Path]:
    """Returns (txt, srt) for a finished track, creating them on the first request. `channel` is the one that stands for the raw recording, if not both."""
    txt, srt = folder / "transcript.txt", folder / "transcript.srt"
    # One at a time: Whisper is CPU heavy and the request may arrive twice from two buttons.
    with _lock:
        if txt.exists() and srt.exists():
            return txt, srt
        global _model
        try:
            if _model is None:
                from faster_whisper import WhisperModel

                _model = WhisperModel(MODEL, device="cpu", compute_type="int8")
            # Channel 2 is the enhanced stem: cleaner speech gives a more accurate transcript, unless noise buried the voice.
            # There Whisper reads the raw recording too; its own confidence picked the better of the two on all nine recordings tested.
            segments = _read(folder, "c2")
            if speech_share < SUSPECT_BELOW_DB:
                recording = _read(folder, "0.5*c0+0.5*c1" if channel is None else f"c{channel}")
                if _confidence(recording) > _confidence(segments):
                    segments = recording
            lines, cues = [], []
            for i, segment in enumerate(segments, 1):
                text = segment.text.strip()
                lines.append(text)
                cues.append(f"{i}\n{_timestamp(segment.start)} --> {_timestamp(segment.end)}\n{text}\n")
            txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
            srt.write_text("\n".join(cues), encoding="utf-8")
        finally:
            # Transcripts are cached, so the model is only needed again for another file.
            _model = None
            engines.release_memory()
    return txt, srt
