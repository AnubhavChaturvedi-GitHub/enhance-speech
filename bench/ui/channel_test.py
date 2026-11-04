"""Checks that real Chrome delivers each channel of the merged file to the right splitter output."""
import asyncio, json, subprocess, sys
from pathlib import Path
import numpy as np
from playwright.async_api import async_playwright
ROOT = Path(__file__).resolve().parents[2]
web = ROOT / "web" / "assets"
FREQS = [300, 500, 700, 900, 1100, 1300, 1500, 1700]
sr = 48000
t = np.arange(sr * 6) / sr
x = np.stack([0.2 * np.sin(2 * np.pi * f * t) for f in FREQS], axis=1).astype(np.float32)
raw = web / "_chan.raw"
x.tofile(raw)
out = web / "_chan.mp4"
subprocess.run(["ffmpeg","-y","-v","error","-f","f32le","-ar","48000","-ac","8","-channel_layout","7.1","-i",str(raw),"-c:a","flac","-sample_fmt","s32","-strict","-2","-movflags","+faststart",str(out)],check=True)
raw.unlink()
JS = """
async () => {
  const FREQS = %s;
  const el = document.createElement(window.__tag);
  el.crossOrigin='anonymous'; el.src='assets/_chan.mp4'; el.muted=false; el.loop=true;
  await new Promise((res,rej)=>{el.oncanplay=res; el.onerror=()=>rej(new Error('media error '+(el.error&&el.error.message))); el.load();});
  const ctx = new AudioContext();
  const src = ctx.createMediaElementSource(el);
  src.channelInterpretation='discrete';
  const sp = ctx.createChannelSplitter(8);
  sp.channelInterpretation='discrete'; sp.channelCountMode='explicit';
  src.connect(sp);
  const an=[]; const sink=ctx.createGain(); sink.gain.value=0; sink.connect(ctx.destination);
  for(let i=0;i<8;i++){const a=ctx.createAnalyser();a.fftSize=8192;a.smoothingTimeConstant=0;sp.connect(a,i);a.connect(sink);an.push(a);}
  await ctx.resume(); await el.play();
  await new Promise(r=>setTimeout(r,2500));
  const out=[];
  for(let i=0;i<8;i++){
    const d=new Float32Array(an[i].frequencyBinCount); an[i].getFloatFrequencyData(d);
    const hz=ctx.sampleRate/an[i].fftSize; let best=-1,bv=-1e9;
    for(let k=2;k<d.length;k++){ if(d[k]>bv){bv=d[k];best=k;} }
    out.push({channel:i, peakHz:Math.round(best*hz), dB:Math.round(bv)});
  }
  el.pause();
  return {sampleRate:ctx.sampleRate, channels:out};
}
""" % json.dumps(FREQS)
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless=True, args=["--autoplay-policy=no-user-gesture-required"])
        page = await b.new_page()
        await page.goto("http://127.0.0.1:5840/")
        for tag in ["audio","video"]:
            await page.evaluate(f"window.__tag='{tag}'")
            r = await page.evaluate(JS)
            print(tag, "sampleRate", r["sampleRate"])
            ok = True
            for c, f in zip(r["channels"], FREQS):
                good = abs(c["peakHz"]-f) < 12
                ok &= good
                print(f"  splitter {c['channel']}: heard {c['peakHz']} Hz at {c['dB']} dB, sent {f} Hz {'OK' if good else 'WRONG'}")
            print("  ALL OK" if ok else "  MISMATCH")
        await b.close()
    out.unlink()
asyncio.run(main())
