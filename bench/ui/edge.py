"""Edge cases through the real server: tiny, silent, ProRes and rotated files, and cross-site requests."""
import os, json, subprocess, time, urllib.request, urllib.error
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "bench" / "real"
URL = os.environ.get("ENHANCE_URL", "http://127.0.0.1:5840")

def post(path, mode="v2"):
    boundary = "----e2eboundary"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"modelVersion\"\r\n\r\n{mode}\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(URL + "/api/tracks", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.load(urllib.request.urlopen(req))

def wait(track_id, limit=300):
    start = time.time()
    while time.time() - start < limit:
        tracks = json.load(urllib.request.urlopen(URL + "/api/tracks"))
        t = next((x for x in tracks if x["id"] == track_id), None)
        if t and t["state"] in {"finished", "failed"}:
            return t
        time.sleep(1)
    raise TimeoutError(track_id)

def probe(url_path, out):
    urllib.request.urlretrieve(URL + url_path, out)
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,channels,width,height:stream_side_data=rotation:format=duration", "-of", "json", str(out)], capture_output=True, text=True)
    return json.loads(r.stdout)

tmp = Path(__file__).parent / "downloads"
tmp.mkdir(exist_ok=True)
for name in ["short.wav", "silent.wav", "prores.mov", "rotated.mp4"]:
    for mode in (["v2", "v1"] if name in {"short.wav", "silent.wav"} else ["v2"]):
        t0 = time.time()
        tr = post(REAL / name, mode)
        t = wait(tr["id"])
        print(f"{name} [{mode}] -> {t['state']} in {time.time()-t0:.0f}s duration={t['duration']} video={t['isVideo']} error={t['error']}")
        if t["state"] == "finished":
            m = probe(f"/api/tracks/{t['id']}/media", tmp / f"_m_{mode}_{name}.mp4")
            print("   preview:", [(s.get("codec_name"), s.get("channels"), s.get("width"), s.get("height"), s.get("side_data_list")) for s in m["streams"]])
            ready = json.load(urllib.request.urlopen(URL + f"/api/tracks/{t['id']}/export?speech=80"))
            e = probe(ready["url"], tmp / f"_e_{mode}_{name}")
            print("   export :", [(s.get("codec_name"), s.get("channels"), s.get("width"), s.get("height"), s.get("side_data_list")) for s in e["streams"]], e["format"]["duration"])
        urllib.request.urlopen(urllib.request.Request(URL + f"/api/tracks/{tr['id']}", method="DELETE"))

# Another website trying to upload or delete through the visitor's browser must be refused.
req = urllib.request.Request(URL + "/api/tracks/abc", method="DELETE", headers={"Origin": "https://evil.example"})
try:
    urllib.request.urlopen(req); print("cross-site delete: NOT refused")
except urllib.error.HTTPError as e:
    print("cross-site delete refused with", e.code)
req = urllib.request.Request(URL + "/api/tracks", headers={"Host": "evil.example"})
try:
    urllib.request.urlopen(req); print("foreign host: NOT refused")
except urllib.error.HTTPError as e:
    print("foreign host refused with", e.code)
