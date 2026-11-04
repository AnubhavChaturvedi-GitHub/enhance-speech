"""Uploads a long file and tracks server memory and speed. Run seam_check.py on the result for the block seams."""
import os, json, subprocess, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
URL = os.environ.get("ENHANCE_URL", "http://127.0.0.1:5840")
src = ROOT / "bench/real" / (__import__("sys").argv[1] if len(__import__("sys").argv) > 1 else "long.wav")
boundary = "----e2eboundary"
body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"modelVersion\"\r\n\r\nv2\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{src.name}\"\r\nContent-Type: audio/wav\r\n\r\n").encode() + src.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
track = json.load(urllib.request.urlopen(urllib.request.Request(URL + "/api/tracks", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})))
del body
pid = subprocess.run("pgrep -f 'uvicorn server.app' | head -1", shell=True, capture_output=True, text=True).stdout.strip()
t0 = time.time(); peak = 0; last = None
while True:
    rss = int(subprocess.run(["ps", "-o", "rss=", "-p", pid], capture_output=True, text=True).stdout.strip() or 0) / 1024
    peak = max(peak, rss)
    t = next(x for x in json.load(urllib.request.urlopen(URL + "/api/tracks")) if x["id"] == track["id"])
    if t["stage"] != last:
        print(f"{time.time()-t0:7.0f}s  stage {t['stage']}  rss {rss:.0f} MB", flush=True); last = t["stage"]
    if t["state"] in {"finished", "failed"}:
        break
    time.sleep(3)
dt = time.time() - t0
# A failed job has no duration, and its error is the one thing this line must not lose.
duration = t["duration"] or 0
print(f"{t['state']} error={t['error']} duration={duration:.1f}s took {dt:.0f}s = {duration/dt:.2f}x realtime; peak server memory {peak:.0f} MB")
print("TRACK", track["id"])
