"""Track store and the single-worker queue. Files process one at a time, in upload order."""
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from . import engines, media, pipeline

log = logging.getLogger("enhance")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TRACKS = DATA / "tracks"
INDEX = DATA / "tracks.json"

# Share of the whole job each stage takes, for the progress ring.
WEIGHTS = {
    "v2": {"processing": 0.02, "separating_background": 0.75, "enhancing": 0.20, "removing_echo": 0.01, "finishing": 0.02},
    "v1": {"processing": 0.05, "removing_echo": 0.10, "enhancing": 0.70, "finishing": 0.15},
}
STAGE_ORDER = {
    "v2": ["processing", "separating_background", "enhancing", "removing_echo", "finishing"],
    "v1": ["processing", "removing_echo", "enhancing", "finishing"],
}


class Cancelled(Exception):
    pass


class Store:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.tracks: dict[str, dict] = {}
        TRACKS.mkdir(parents=True, exist_ok=True)
        self._load()
        threading.Thread(target=self._worker, daemon=True, name="enhance-worker").start()

    def _load(self) -> None:
        if INDEX.exists():
            self.tracks = json.loads(INDEX.read_text())
        for track_id, track in list(self.tracks.items()):
            # A job interrupted by a restart goes back in the queue.
            if track["state"] == "processing":
                track.update(state="queued", stage=None, progress=0)
            # An upload interrupted by a restart never finished arriving.
            elif track["state"] == "receiving":
                del self.tracks[track_id]
                shutil.rmtree(TRACKS / track_id, ignore_errors=True)
        self._save()
        self.wake.set()

    def _save(self) -> None:
        tmp = INDEX.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.tracks, indent=1))
        tmp.replace(INDEX)

    def list(self) -> list[dict]:
        with self.lock:
            return sorted((dict(t) for t in self.tracks.values()), key=lambda t: t["createdAt"], reverse=True)

    def get(self, track_id: str) -> dict | None:
        with self.lock:
            track = self.tracks.get(track_id)
            return dict(track) if track else None

    def create(self, name: str, mode: str, source_name: str) -> tuple[str, Path]:
        track_id = uuid.uuid4().hex[:16]
        folder = TRACKS / track_id
        folder.mkdir(parents=True)
        with self.lock:
            self.tracks[track_id] = {
                "id": track_id,
                "trackName": name,
                "sourceName": source_name,
                "modelVersion": mode,
                # Not eligible for the worker until the file has fully arrived.
                "state": "receiving",
                "stage": None,
                "progress": 0,
                "duration": None,
                "isVideo": False,
                "error": None,
                "createdAt": time.time(),
            }
            self._save()
        return track_id, folder

    def enqueue(self, track_id: str) -> None:
        self._update(track_id, state="queued")
        self.wake.set()

    def delete(self, track_id: str) -> bool:
        with self.lock:
            if track_id not in self.tracks:
                return False
            del self.tracks[track_id]
            self._save()
        # A running job notices the missing track at its next progress report.
        shutil.rmtree(TRACKS / track_id, ignore_errors=True)
        return True

    def _update(self, track_id: str, **fields) -> None:
        with self.lock:
            if track_id in self.tracks:
                self.tracks[track_id].update(fields)
                self._save()

    def _next(self) -> dict | None:
        with self.lock:
            queued = [t for t in self.tracks.values() if t["state"] == "queued"]
            return dict(min(queued, key=lambda t: t["createdAt"])) if queued else None

    def _worker(self) -> None:
        while True:
            job = self._next()
            if job is None:
                try:
                    engines.release_models()
                except Exception:  # noqa: BLE001
                    # Models left in memory only cost memory; a dead worker would stop every later file.
                    log.error("could not release the models\n%s", traceback.format_exc())
                self.wake.wait()
                self.wake.clear()
                continue
            try:
                self._run(job)
            except (Exception, SystemExit):  # noqa: BLE001
                # A full disk can stop the job's failure being saved; a dead worker would stop every later file.
                log.error("job could not finish\n%s", traceback.format_exc())

    def _run(self, job: dict) -> None:
        track_id, mode = job["id"], job["modelVersion"]
        folder = TRACKS / track_id
        source = folder / job["sourceName"]
        work = folder / "work"
        order, weights = STAGE_ORDER[mode], WEIGHTS[mode]
        last_write = [0.0]

        def report(stage: str, fraction: float) -> None:
            if self.get(track_id) is None:
                raise Cancelled()
            done = sum(weights[s] for s in order[: order.index(stage)]) + weights[stage] * min(1.0, fraction)
            # Progress arrives many times a second; the index only needs a few writes.
            now = time.time()
            if now - last_write[0] > 0.5 or fraction >= 1.0:
                last_write[0] = now
                self._update(track_id, state="processing", stage=stage, progress=round(done, 4))

        try:
            self._update(track_id, state="processing", stage="processing", progress=0)
            meta = pipeline.process(source, work, folder / "merged.mp4", mode, report)
            self._update(track_id, state="finished", stage=None, progress=1, error=None, **meta)
        except Cancelled:
            log.info("cancelled %s", track_id)
        except media.MediaError as err:
            self._update(track_id, state="failed", stage=None, error=str(err))
        except (Exception, SystemExit):  # noqa: BLE001
            # Deleting a running track pulls its folder away, which can surface here as a file error. DeepFilterNet calls exit() on a broken model cache.
            if self.get(track_id) is not None:
                log.error("job failed\n%s", traceback.format_exc())
                self._update(track_id, state="failed", stage=None, error="There was an unknown problem enhancing the file.")
        finally:
            shutil.rmtree(work, ignore_errors=True)
            # A delete can lose the race with files being created in work; the worker has let go of the folder by now.
            if self.get(track_id) is None:
                shutil.rmtree(folder, ignore_errors=True)
