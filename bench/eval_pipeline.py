"""Runs the real pipeline on one file and unpacks its stems for scoring."""
import sys, time, shutil, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server import pipeline

src = Path(sys.argv[1]); tag = sys.argv[2]; mode = sys.argv[3] if len(sys.argv) > 3 else "v2"
# The tag names a folder that gets wiped, so it must stay a plain name inside bench/work.
if not tag or tag in {".", ".."} or Path(tag).name != tag:
    sys.exit("usage: eval_pipeline.py <file> <tag> [v1|v2]")
work = ROOT / "bench" / "work" / tag
shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
out = work / "merged.mp4"
marks = {}
t0 = time.time()
def on_stage(stage, f):
    marks.setdefault(stage, time.time())
    print(f"\r{stage:<24}{f*100:5.1f}%  {time.time()-t0:6.1f}s", end="", flush=True)
meta = pipeline.process(src, work, out, mode, on_stage)
dt = time.time() - t0
print(f"\nmeta {meta}\ntotal {dt:.1f}s  speed {meta['duration']/dt:.2f}x realtime")
keys = list(marks)
for a, b in zip(keys, keys[1:] + [None]):
    print(f"  {a:<24}{(marks[b] if b else t0+dt) - marks[a]:6.1f}s")
print("merged size MB", out.stat().st_size / 1e6)
