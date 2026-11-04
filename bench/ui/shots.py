"""Screenshots of the app in real headless Chrome. Usage: shots.py <name>:<hash>[:<width>x<height>] ..."""
import asyncio, sys
from pathlib import Path
from playwright.async_api import async_playwright
OUT = Path(__file__).parent / "shots"
OUT.mkdir(exist_ok=True)
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless=True, args=["--autoplay-policy=no-user-gesture-required"])
        for spec in sys.argv[1:]:
            parts = spec.split(":")
            name, h = parts[0], parts[1]
            w, ht = (int(v) for v in (parts[2] if len(parts) > 2 else "1440x900").split("x"))
            full = len(parts) > 3 and parts[3] == "full"
            ctx = await b.new_context(viewport={"width": w, "height": ht})
            page = await ctx.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            await page.goto(f"http://127.0.0.1:5840/#{h}")
            await page.wait_for_timeout(1500)
            await page.screenshot(path=str(OUT / f"{name}.png"), full_page=full)
            print(name, "errors:", errors[:5])
            await ctx.close()
        await b.close()
asyncio.run(main())
