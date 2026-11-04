"""Captures the workspace picture for the landing page, using only the public-domain demo sample."""
import asyncio, shutil, subprocess, sys
from pathlib import Path
from playwright.async_api import async_playwright
ROOT = Path(__file__).resolve().parents[2]
TMP = Path(__file__).parent / "mock"
TMP.mkdir(exist_ok=True)
names = ["episode-12-interview.mp3", "conversation-with-priya.mp3", "guest-intro.mp3"]
src = ROOT / "web" / "assets" / "sample" / "original.mp3"
for n in names:
    shutil.copy(src, TMP / n)
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless=True)
        ctx = await b.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page = await ctx.new_page()
        await page.goto("http://127.0.0.1:5840/#/enhance")
        await page.wait_for_timeout(1000)
        # Other tracks would show in the picture, and they may be real work with no undo, so this never deletes them.
        if await page.evaluate("fetch('/api/tracks').then(r=>r.json())"):
            sys.exit("The server still holds tracks. Empty the list yourself before taking this picture.")
        await page.set_input_files("#file-input", [str(TMP / n) for n in names])
        for _ in range(900):
            tracks = await page.evaluate("fetch('/api/tracks').then(r=>r.json())")
            if tracks and any(t["trackName"] == names[0] and t["state"] == "finished" for t in tracks) and any(t["state"] == "processing" for t in tracks):
                break
            await page.wait_for_timeout(1000)
        await page.click(f'[data-track]:has-text("{names[0]}")')
        await page.wait_for_selector("#play")
        await page.wait_for_timeout(1500)
        await page.evaluate("document.querySelector('.action-bar').style.display='none'")
        await page.screenshot(path=str(TMP / "workspace.png"), clip={"x": 0, "y": 88, "width": 1440, "height": 640})
        for t in await page.evaluate("fetch('/api/tracks').then(r=>r.json())"):
            if t["trackName"] in names:
                await page.evaluate(f"fetch('/api/tracks/{t['id']}',{{method:'DELETE'}})")
        await b.close()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(TMP / "workspace.png"), "-q:v", "3", str(ROOT / "web" / "assets" / "workspace.jpg")], check=True)
asyncio.run(main())
