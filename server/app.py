"""Local HTTP server: uploads, track list, stem media for the player, and exports."""
from __future__ import annotations

import errno
import logging
import re
import shutil
from pathlib import Path
from urllib.parse import quote, urlparse

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import dsp, jobs, media, transcribe

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"

AUDIO_EXT = {"wav", "mp3", "m4a", "aac", "flac", "ogg", "opus", "aiff", "aif", "wma", "webm"}
VIDEO_EXT = {"mp4", "mov", "m4v", "mkv", "avi", "webm", "3gp"}

app = FastAPI(title="Enhance Speech")
# The server only answers to its own local address, so other websites open in the browser cannot
# reach it by pointing a hostname at this machine.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"])
store = jobs.Store()


@app.middleware("http")
async def same_origin_writes(request: Request, call_next):
    """Stops other websites from uploading or deleting files through the visitor's browser."""
    origin = request.headers.get("origin")
    if request.method not in {"GET", "HEAD"} and origin:
        host = urlparse(origin).hostname
        if host not in {"localhost", "127.0.0.1"}:
            return JSONResponse({"detail": "Cross-site request refused"}, status_code=403)
    return await call_next(request)


def _public(track: dict) -> dict:
    """What the browser needs to know about a track."""
    return {
        "id": track["id"],
        "trackName": track["trackName"],
        "modelVersion": track["modelVersion"],
        "state": "queued" if track["state"] == "receiving" else track["state"],
        "stage": track["stage"],
        "progress": track["progress"],
        "duration": track["duration"],
        "isVideo": track["isVideo"],
        "error": track["error"],
        "createdAt": track["createdAt"],
    }


def _require(track_id: str) -> dict:
    track = store.get(track_id)
    if track is None:
        raise HTTPException(404, "Track not found")
    return track


@app.get("/api/tracks")
def list_tracks() -> list[dict]:
    return [_public(t) for t in store.list()]


@app.post("/api/tracks")
def create_track(file: UploadFile = File(...), modelVersion: str = Form("v2")) -> dict:
    name = Path(file.filename or "audio").name
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in AUDIO_EXT | VIDEO_EXT:
        raise HTTPException(415, "File extension or content not currently supported for enhancement.")
    mode = "v1" if modelVersion == "v1" else "v2"
    track_id = None
    try:
        track_id, folder = store.create(name, mode, f"source.{ext}")
        with open(folder / f"source.{ext}", "wb") as out:
            shutil.copyfileobj(file.file, out, 1024 * 1024)
        store.enqueue(track_id)
    except OSError as err:
        # No row may stay behind holding part of a file.
        if track_id is not None:
            store.delete(track_id)
        if err.errno == errno.ENOSPC:
            raise HTTPException(507, "There is not enough free disk space for this file.") from err
        raise
    track = store.get(track_id)
    # Deleted from the list while it was still arriving.
    if track is None:
        raise HTTPException(404, "Track not found")
    return _public(track)


@app.delete("/api/tracks/{track_id}")
def delete_track(track_id: str) -> dict:
    if not store.delete(track_id):
        raise HTTPException(404, "Track not found")
    return {"ok": True}


@app.get("/api/tracks/{track_id}/media")
def track_media(track_id: str) -> FileResponse:
    _require(track_id)
    path = jobs.TRACKS / track_id / "merged.mp4"
    if not path.exists():
        raise HTTPException(404, "Not ready")
    return FileResponse(path, media_type="video/mp4")


def _safe_stem(name: str) -> str:
    base = name.rsplit(".", 1)[0] if "." in name else name
    clean = re.sub(r'[\\/:*?"<>|]+', "_", base)
    # Room is left for what an export adds to the name; a filesystem takes about 255 UTF-16 units.
    return clean.encode("utf-16-le", "ignore")[:400].decode("utf-16-le", "ignore").strip() or "audio"


@app.get("/api/tracks/{track_id}/export")
def export_track(
    track_id: str,
    speech: float = 50,
    speechMuted: bool = False,
    background: float = 10,
    backgroundMuted: bool = False,
    music: float = 10,
    musicMuted: bool = False,
    enhanced: bool = True,
    component: str = "full",
) -> dict:
    """Builds the requested mix and says where to fetch it.

    Preparing first lets the page show a spinner during the mix, then hand the finished file to the
    browser's own download, which goes straight to disk however large it is.
    """
    track = _require(track_id)
    if track["state"] != "finished":
        raise HTTPException(409, "Not ready")
    folder = jobs.TRACKS / track_id
    merged = folder / "merged.mp4"
    source = folder / track["sourceName"]
    stem = _safe_stem(track["trackName"])
    source_ext = track["sourceName"].rsplit(".", 1)[-1]
    exports = folder / "exports"
    exports.mkdir(exist_ok=True)

    # With Enhance Speech switched off, the mix is simply the untouched original.
    if component == "full" and not enhanced:
        out = exports / f"{stem}.{source_ext}"
        if not out.exists():
            out.hardlink_to(source)
        return {"url": f"/api/tracks/{track_id}/files/{quote(out.name)}"}

    gains = dsp.download_gains(speech, speechMuted, background, backgroundMuted, music, musicMuted, track["modelVersion"] == "v1")
    only = {
        "speech": {"original": 0.0, "enhanced": gains["enhanced"], "isolated": gains["isolated"], "music": 0.0, "background": 0.0},
        "background": {"original": 0.0, "enhanced": 0.0, "isolated": 0.0, "music": 0.0, "background": gains["background"]},
        "music": {"original": 0.0, "enhanced": 0.0, "isolated": 0.0, "music": gains["music"], "background": 0.0},
    }
    try:
        if component in only:
            out = exports / f"{stem} ({component}).wav"
            media.export_audio(merged, only[component], out)
        elif track["isVideo"]:
            ext = source_ext if source_ext in {"mp4", "mov", "m4v"} else "mp4"
            out = exports / f"{stem} (enhanced).{ext}"
            media.export_video(merged, source, gains, out)
        else:
            out = exports / f"{stem} (enhanced).wav"
            media.export_audio(merged, gains, out)
    except media.MediaError as err:
        raise HTTPException(500, str(err)) from err
    return {"url": f"/api/tracks/{track_id}/files/{quote(out.name)}"}


@app.get("/api/tracks/{track_id}/files/{name}")
def exported_file(track_id: str, name: str) -> FileResponse:
    _require(track_id)
    # The name comes from the URL, so it must stay inside this track's exports folder.
    path = jobs.TRACKS / track_id / "exports" / Path(name).name
    if not path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(path, filename=path.name)


# HEAD lets the page wait for Whisper to finish, then start the browser's own download.
@app.api_route("/api/tracks/{track_id}/transcript", methods=["GET", "HEAD"])
def track_transcript(track_id: str, format: str = "txt") -> FileResponse:
    """Plain transcript (txt) or captions (srt), made on the first request and cached."""
    track = _require(track_id)
    if track["state"] != "finished":
        raise HTTPException(409, "Not ready")
    if format not in {"txt", "srt"}:
        raise HTTPException(400, "Unknown format")
    folder = jobs.TRACKS / track_id
    try:
        txt, srt = transcribe.transcript_files(folder, track.get("speechShare", 0.0), track.get("rawChannel"))
    except media.MediaError as err:
        raise HTTPException(500, str(err)) from err
    path = txt if format == "txt" else srt
    return FileResponse(path, filename=f"{_safe_stem(track['trackName'])}.{format}", media_type="text/plain; charset=utf-8")


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
