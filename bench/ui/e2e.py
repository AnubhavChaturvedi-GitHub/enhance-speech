"""End-to-end test through the real UI in real Chrome (headless, temporary profile)."""
import os, asyncio, json, subprocess, sys, time
from pathlib import Path
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "bench" / "real"
SHOTS = Path(__file__).parent / "shots"
DOWN = Path(__file__).parent / "downloads"
SHOTS.mkdir(exist_ok=True)
DOWN.mkdir(exist_ok=True)
URL = os.environ.get("ENHANCE_URL", "http://127.0.0.1:5840")
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


# Taps the graph so the test can read real gain values and measure what would reach the speakers.
TAP = """
(() => {
  const realMerger = AudioContext.prototype.createChannelMerger;
  AudioContext.prototype.createChannelMerger = function (n) {
    const m = realMerger.call(this, n);
    const a = this.createAnalyser();
    a.fftSize = 8192;
    const realConnect = m.connect;
    m.connect = function (dest, ...rest) {
      window.__tap = a;
      window.__ctx = this.context;
      realConnect.call(m, a);
      return realConnect.call(m, dest, ...rest);
    };
    return m;
  };
  window.__level = async (ms = 700) => {
    const a = window.__tap;
    if (!a) return null;
    const d = new Float32Array(a.fftSize);
    let peak = 0, sum = 0, n = 0;
    const end = performance.now() + ms;
    while (performance.now() < end) {
      a.getFloatTimeDomainData(d);
      for (const v of d) { sum += v * v; n++; peak = Math.max(peak, Math.abs(v)); }
      await new Promise((r) => setTimeout(r, 40));
    }
    return { rmsDb: 10 * Math.log10(sum / n + 1e-12), peak };
  };
})();
"""


async def wait_state(page, name, states, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        tracks = await page.evaluate("fetch('/api/tracks').then(r=>r.json())")
        for t in tracks:
            if t["trackName"] == name and t["state"] in states:
                return t
        await page.wait_for_timeout(1000)
    raise TimeoutError(f"{name} never reached {states}")


async def set_slider(page, key, percent):
    box = await page.locator(f'[data-area="{key}"] .slider').bounding_box()
    await page.mouse.click(box["x"] + box["width"] * percent / 100, box["y"] + 12)
    await page.wait_for_timeout(250)


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,channels,sample_rate:format=duration", "-of", "json", str(path)], capture_output=True, text=True)
    return json.loads(out.stdout)


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=True,
            args=["--autoplay-policy=no-user-gesture-required"],
        )
        ctx = await browser.new_context(viewport={"width": 1440, "height": 900}, accept_downloads=True)
        await ctx.add_init_script(TAP)
        page = await ctx.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

        # Landing and the sample toggle
        await page.goto(URL + "/#/")
        await page.wait_for_timeout(800)
        await page.click("#sample-play")
        await page.wait_for_timeout(600)
        check("hero sample starts playing", "Pause" in (await page.get_attribute("#sample-play", "aria-label")))
        await page.click("#sample-switch")
        check("hero switch flips", (await page.get_attribute("#sample-switch", "aria-checked")) == "true")
        await page.click("#sample-play")
        await page.screenshot(path=str(SHOTS / "e2e_landing.png"))

        # Upload one audio file through the real input
        await page.goto(URL + "/#/enhance")
        await page.wait_for_timeout(500)
        await page.set_input_files("#file-input", str(REAL / "nomic60.wav"))
        await page.wait_for_selector('[data-track] .track-title:has-text("nomic60.wav")', timeout=15000)
        check("upload appears in list", True)
        t = await wait_state(page, "nomic60.wav", {"processing"}, 60)
        await page.wait_for_timeout(2500)
        await page.screenshot(path=str(SHOTS / "e2e_processing.png"))
        check("processing view shows", await page.locator(".processing-detail").count() == 1, await page.locator(".processing-text").inner_text())

        # Queue a second file while the first runs
        await page.set_input_files("#file-input", str(REAL / "bgm40.wav"))
        await wait_state(page, "bgm40.wav", {"queued"}, 30)
        await page.wait_for_timeout(1600)
        await page.screenshot(path=str(SHOTS / "e2e_queue.png"))
        check("second file is queued", True)

        start = time.time()
        t = await wait_state(page, "nomic60.wav", {"finished", "failed"})
        took = time.time() - start
        check("first file finished", t["state"] == "finished", f"{t.get('error')} duration {t['duration']:.1f}s")
        await page.wait_for_selector("#play", timeout=20000)
        await page.wait_for_timeout(1500)
        await page.screenshot(path=str(SHOTS / "e2e_detail.png"))
        labels = await page.locator(".strength-value").all_inner_texts()
        check("defaults match reference tool (speech/music/background)", labels == ["50%", "10%", "10%"], str(labels))

        # Playback and mixing
        await page.click("#play")
        await page.wait_for_timeout(1500)
        now = await page.inner_text("#now")
        level = await page.evaluate("window.__level()")
        check("audio plays and is audible", level and level["rmsDb"] > -45, f"{level}, time {now}")
        check("time advances", now != "0:00", now)

        await set_slider(page, "speech", 99)
        check("speech slider label follows", (await page.locator('[data-card="speech"] .strength-value').inner_text()) in {"99%", "100%", "98%"})
        await page.click('[data-mute="speech"]')
        await page.wait_for_timeout(200)
        await set_slider(page, "background", 0)
        await set_slider(page, "music", 0)
        await page.wait_for_timeout(400)
        muted = await page.evaluate("window.__level()")
        check("muting everything is silent", muted and muted["rmsDb"] < -70, str(muted))
        await page.click('[data-mute="speech"]')
        await page.wait_for_timeout(400)
        speech_only = await page.evaluate("window.__level()")
        check("unmuting speech brings it back", speech_only and speech_only["rmsDb"] > -45, str(speech_only))
        await page.click("#enhance-switch")
        await page.wait_for_timeout(400)
        off = await page.evaluate("window.__level()")
        check("toggle off plays the original", off and off["rmsDb"] > -55, str(off))
        check("sliders disabled when off", await page.locator('[data-card="speech"].disabled').count() == 1)
        await page.click("#enhance-switch")
        box = await page.locator("#scrubber").bounding_box()
        await page.mouse.click(box["x"] + box["width"] * 0.7, box["y"] + 7)
        await page.wait_for_timeout(700)
        check("seek moves the playhead", (await page.inner_text("#now")) >= "0:2", await page.inner_text("#now"))
        await page.click("#play")
        await set_slider(page, "speech", 75)
        await set_slider(page, "background", 30)
        await set_slider(page, "music", 20)

        # Downloads
        async with page.expect_download(timeout=120000) as dl:
            await page.click("#download")
        d = await dl.value
        full = DOWN / d.suggested_filename
        await d.save_as(full)
        info = probe(full)
        check("download is named and stereo wav", d.suggested_filename == "nomic60 (enhanced).wav" and info["streams"][0]["channels"] == 2 and info["streams"][0]["codec_name"].startswith("pcm"), f"{d.suggested_filename} {info['streams'][0]}")
        check("download length matches", abs(float(info["format"]["duration"]) - t["duration"]) < 0.1, info["format"]["duration"])
        for key in ["speech", "music", "background"]:
            async with page.expect_download(timeout=120000) as dl:
                await page.click(f'[data-stem="{key}"]')
            d = await dl.value
            await d.save_as(DOWN / d.suggested_filename)
            check(f"{key} stem download", d.suggested_filename == f"nomic60 ({key}).wav", d.suggested_filename)
        async with page.expect_download(timeout=300000) as dl:
            await page.click("#get-transcript")
        d = await dl.value
        await d.save_as(DOWN / d.suggested_filename)
        text = (DOWN / d.suggested_filename).read_text()
        check("transcript download", "code" in text.lower(), text[:90].replace("\n", " "))
        async with page.expect_download(timeout=120000) as dl:
            await page.click("#get-captions")
        d = await dl.value
        await d.save_as(DOWN / d.suggested_filename)
        check("captions download", "-->" in (DOWN / d.suggested_filename).read_text(), d.suggested_filename)

        # Queue carries on; select the running one and come back
        t2 = await wait_state(page, "bgm40.wav", {"finished", "failed"})
        check("queued file then finished", t2["state"] == "finished", t2.get("error") or "")
        await page.click('[data-track]:has-text("bgm40.wav")')
        await page.wait_for_selector("#play", timeout=20000)
        await page.wait_for_timeout(800)
        await page.screenshot(path=str(SHOTS / "e2e_second.png"))

        # Video
        await page.set_input_files("#file-input", str(REAL / "reel22.mp4"))
        tv = await wait_state(page, "reel22.mp4", {"finished", "failed"})
        check("video finished", tv["state"] == "finished" and tv["isVideo"], tv.get("error") or "")
        await page.click('[data-track]:has-text("reel22.mp4")')
        await page.wait_for_selector("#video-box video", timeout=20000)
        await page.wait_for_timeout(1500)
        await page.click("#play")
        await page.wait_for_timeout(2000)
        state = await page.evaluate("(()=>{const v=document.querySelector('#video-box video');return {t:v.currentTime,w:v.videoWidth,h:v.videoHeight,err:v.error&&v.error.message,ready:v.readyState}})()")
        level = await page.evaluate("window.__level()")
        check("video plays with mixed audio", state["t"] > 0.3 and state["w"] > 0 and level and level["rmsDb"] > -50, f"{state} {level}")
        await page.screenshot(path=str(SHOTS / "e2e_video.png"))
        await page.click("#play")
        async with page.expect_download(timeout=120000) as dl:
            await page.click("#download")
        d = await dl.value
        await d.save_as(DOWN / d.suggested_filename)
        info = probe(DOWN / d.suggested_filename)
        kinds = {s["codec_type"]: s["codec_name"] for s in info["streams"]}
        check("video export keeps the video and adds audio", kinds.get("video") == "hevc" and kinds.get("audio") == "aac", f"{d.suggested_filename} {kinds}")

        # Failure cases
        await page.set_input_files("#file-input", str(REAL / "no_audio.mp4"))
        tn = await wait_state(page, "no_audio.mp4", {"finished", "failed"}, 60)
        check("video without audio fails cleanly", tn["state"] == "failed" and "no audio" in (tn["error"] or ""), tn["error"])
        await page.set_input_files("#file-input", str(REAL / "corrupt.mp3"))
        tc = await wait_state(page, "corrupt.mp3", {"finished", "failed"}, 60)
        check("corrupt file fails cleanly", tc["state"] == "failed", tc["error"])
        await page.wait_for_timeout(1700)
        await page.screenshot(path=str(SHOTS / "e2e_failed.png"))
        bad = REAL / "notes.txt"
        bad.write_text("hello")
        await page.set_input_files("#file-input", str(bad))
        await page.wait_for_selector(".toast", timeout=8000)
        check("unsupported type shows a message", "not currently supported" in await page.locator(".toast").first.inner_text())
        bad.unlink()

        # Quick v1 mode
        await page.select_option("#version", "v1")
        await page.set_input_files("#file-input", str(REAL / "nomic60.wav"))
        await page.wait_for_timeout(1500)
        tracks = await page.evaluate("fetch('/api/tracks').then(r=>r.json())")
        v1_id = next(x["id"] for x in tracks if x["modelVersion"] == "v1")
        start = time.time()
        while True:
            one = next(x for x in await page.evaluate("fetch('/api/tracks').then(r=>r.json())") if x["id"] == v1_id)
            if one["state"] in {"finished", "failed"} or time.time() - start > 400:
                break
            await page.wait_for_timeout(1000)
        check("v1 finished", one["state"] == "finished", f"{one.get('error')} took {time.time()-start:.0f}s")
        await page.click(f'[data-track="{v1_id}"]')
        await page.wait_for_selector("#play", timeout=20000)
        await page.wait_for_timeout(600)
        check("v1 shows one slider", await page.locator(".strength-card").count() == 1)
        await page.screenshot(path=str(SHOTS / "e2e_v1.png"))

        # A long file name must truncate, not push the delete button off the row.
        long_name = REAL / ("a-very-long-recording-name-from-the-field-interview-with-a-guest-" * 3 + ".wav")
        long_name.write_bytes((REAL / "short.wav").read_bytes())
        await page.select_option("#version", "v1")
        await page.set_input_files("#file-input", str(long_name))
        await page.wait_for_selector(f'[data-track] .track-title[title="{long_name.name}"]', timeout=15000)
        row = page.locator(f'[data-track]:has(.track-title[title="{long_name.name}"])')
        row_box = await row.bounding_box()
        del_box = await row.locator(".delete-button").bounding_box()
        check("long name keeps delete button in the row", del_box and del_box["x"] + del_box["width"] <= row_box["x"] + row_box["width"], f"row {row_box['width']:.0f}")
        long_name.unlink()

        # Delete, including while running
        await page.select_option("#version", "v2")
        await page.set_input_files("#file-input", str(REAL / "nomic60.wav"))
        await page.wait_for_timeout(1500)
        tracks = await page.evaluate("fetch('/api/tracks').then(r=>r.json())")
        newest = sorted(tracks, key=lambda x: x["createdAt"])[-1]
        await page.wait_for_timeout(3000)
        await page.click(f'[data-delete="{newest["id"]}"]')
        await page.wait_for_timeout(1500)
        left = await page.evaluate("fetch('/api/tracks').then(r=>r.json())")
        check("delete while processing removes it", all(x["id"] != newest["id"] for x in left))
        check("folder removed", not (ROOT / "data" / "tracks" / newest["id"]).exists())

        # Survives a reload
        await page.reload()
        await page.wait_for_selector("#play", timeout=20000)
        check("files survive a reload", await page.locator("[data-track]").count() >= 3)
        await page.screenshot(path=str(SHOTS / "e2e_reload.png"))

        real_errors = [e for e in errors if "404" not in e and "415" not in e]
        check("no console or page errors", not real_errors, "; ".join(real_errors)[:300])
        await browser.close()
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n{passed}/{len(results)} passed")


asyncio.run(main())
