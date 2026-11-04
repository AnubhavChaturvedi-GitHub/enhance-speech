"""ffmpeg helpers: probing, decoding, building the preview media and exporting mixes."""
from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

SR = 48000
# Video codecs a browser can play, so the preview can copy the stream instead of re-encoding it.
BROWSER_VIDEO = {"h264", "hevc", "vp9", "av1"}


class MediaError(Exception):
    """A problem with the user's file, with a message safe to show them."""


@dataclass
class MediaInfo:
    duration: float
    has_audio: bool
    has_video: bool
    video_codec: str | None
    audio_channels: int


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    # ffmpeg takes commands from its stdin: a "q" typed in the server's terminal would end it early with exit code 0.
    return subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)


def probe(path: Path) -> MediaInfo:
    result = _run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    if result.returncode != 0:
        raise MediaError("The file failed to enhance. It may be incorrectly formatted. Try a different file.")
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    # Cover art in an MP3 or M4A shows up as a one-frame video stream; it is not a video.
    video = next(
        (s for s in streams if s.get("codec_type") == "video" and s.get("disposition", {}).get("attached_pic") != 1),
        None,
    )
    duration = float(data.get("format", {}).get("duration") or (audio or {}).get("duration") or 0)
    return MediaInfo(
        duration=duration,
        has_audio=audio is not None,
        has_video=video is not None,
        video_codec=video.get("codec_name") if video else None,
        audio_channels=int(audio.get("channels", 2)) if audio else 0,
    )


def decode_stereo(source: Path, raw: Path, channels: int) -> None:
    """Decodes the first audio stream to 48 kHz stereo float32 raw, with sample 0 at the file's time 0.

    Phone recordings often start their audio slightly after the video; padding that gap here is what
    keeps the enhanced sound in sync when it is put back with the picture. A mono source goes to both
    channels at its own level; ffmpeg's -ac 2 would take it down 3 dB. The aformat folds any part of a
    file that turns stereo later, which the pan would otherwise cut to its left channel.
    """
    layout = ["-af", "aresample=async=1:first_pts=0,aformat=channel_layouts=mono,pan=stereo|c0=c0|c1=c0"] if channels == 1 else ["-af", "aresample=async=1:first_pts=0", "-ac", "2"]
    result = _run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-map", "0:a:0", *layout, "-ar", str(SR), "-f", "f32le", "-c:a", "pcm_f32le", str(raw)])
    if result.returncode != 0:
        raise MediaError("There was a problem enhancing this file. Please try a different one.")


def build_merged(raw8: Path, source: Path, info: MediaInfo, out: Path) -> None:
    """Muxes the 8-channel stem mix (with the video, if any) into one MP4 for the player.

    One file with every stem in sync lets the browser change the mix instantly while it plays.
    """
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "8", "-channel_layout", "7.1", "-i", str(raw8)]
    if info.has_video:
        cmd += ["-i", str(source), "-map", "1:v:0", "-map", "0:a:0"]
        if info.video_codec in BROWSER_VIDEO:
            cmd += ["-c:v", "copy"]
            if info.video_codec == "hevc":
                cmd += ["-tag:v", "hvc1"]
        else:
            cmd += ["-c:v", "h264_videotoolbox", "-b:v", "6M", "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "flac", "-sample_fmt", "s32", "-compression_level", "5", "-strict", "-2", "-movflags", "+faststart", str(out)]
    result = _run(cmd)
    if result.returncode != 0:
        raise MediaError("There was an unknown problem enhancing the file.")


def decode_for_speech(merged: Path, channels: str, raw: Path) -> None:
    """Pulls a mono mix of the merged file's channels (a pan expression such as "c2") out as 16 kHz float32, the format Whisper wants."""
    result = _run(["ffmpeg", "-y", "-v", "error", "-i", str(merged), "-map", "0:a:0", "-af", f"pan=mono|c0={channels}", "-ar", "16000", "-f", "f32le", "-c:a", "pcm_f32le", str(raw)])
    if result.returncode != 0:
        raise MediaError("There was a problem reading this file.")


def _pan(gains: dict[str, float]) -> str:
    """Stereo pan filter from the merged file's channel order to left and right."""
    o, e, i, m, b = gains["original"], gains["enhanced"], gains["isolated"], gains["music"], gains["background"]
    left = f"{o}*c0+{e}*c2+{i}*c3+{m}*c4+{b}*c6"
    right = f"{o}*c1+{e}*c2+{i}*c3+{m}*c5+{b}*c7"
    return f"pan=stereo|c0={left}|c1={right}"


def _spare_name(out: Path) -> Path:
    """Where an export is written first. The browser may still be fetching the last mix from `out`, which must not change under it."""
    return out.with_name(f".{uuid.uuid4().hex}{out.suffix}")


def export_audio(merged: Path, gains: dict[str, float], out: Path) -> None:
    # The soft limiter keeps a loud mix from clipping once stems are summed.
    af = f"{_pan(gains)},alimiter=limit=0.99:attack=2:release=40:level=disabled:latency=1"
    spare = _spare_name(out)
    try:
        result = _run(["ffmpeg", "-y", "-v", "error", "-i", str(merged), "-map", "0:a:0", "-af", af, "-c:a", "pcm_s24le", str(spare)])
        if result.returncode != 0:
            raise MediaError("An error occurred while downloading the file. Please try again.")
        os.replace(spare, out)
    finally:
        spare.unlink(missing_ok=True)


def export_video(merged: Path, source: Path, gains: dict[str, float], out: Path) -> None:
    """Puts the mix back with the source's video stream, untouched when the output container can hold its codec."""
    af = f"{_pan(gains)},alimiter=limit=0.99:attack=2:release=40:level=disabled:latency=1"
    spare = _spare_name(out)
    base = ["ffmpeg", "-y", "-v", "error", "-i", str(merged), "-i", str(source), "-map", "1:v:0", "-map", "0:a:0", "-af", af]
    tail = ["-c:a", "aac", "-b:a", "320k", "-movflags", "+faststart", "-shortest", str(spare)]
    try:
        result = _run(base + ["-c:v", "copy"] + tail)
        if result.returncode != 0:
            # The target container cannot hold this video codec as it is, so re-encode the picture.
            result = _run(base + ["-c:v", "h264_videotoolbox", "-b:v", "10M", "-pix_fmt", "yuv420p"] + tail)
        if result.returncode != 0:
            raise MediaError("An error occurred while downloading the file. Please try again.")
        os.replace(spare, out)
    finally:
        spare.unlink(missing_ok=True)
