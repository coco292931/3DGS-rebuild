"""实测 pycolmap 的线程/GPU 到底有没有生效。

用同一批图在不同配置下跑特征提取，比耗时。
"""
import shutil
import time
from pathlib import Path

import pycolmap
from pycolmap import FeatureExtractionOptions, FeatureMatchingOptions

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "real4" / "frames"
TMP = ROOT / "output" / "_bench_sfm"
TMP.mkdir(parents=True, exist_ok=True)

frames = sorted(SRC.glob("*.png"))[:60]      # 60 张够看出差别
print(f"用 {len(frames)} 张图测（{frames[0].name} ... {frames[-1].name}）\n")

def bench(tag, *, threads, gpu):
    work = TMP / tag
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    (work / "frames").mkdir(parents=True)
    for f in frames:
        shutil.copy2(f, work / "frames" / f.name)
    db = work / "database.db"
    if db.exists():
        db.unlink()

    opts = FeatureExtractionOptions()
    opts.num_threads = threads
    opts.use_gpu = gpu
    try:
        mode = pycolmap.CameraMode.SINGLE
    except AttributeError:
        mode = "SINGLE"
    t0 = time.time()
    try:
        pycolmap.extract_features(str(db), str(work / "frames"),
                                  camera_mode=mode, extraction_options=opts,
                                  device=pycolmap.Device.auto)
        dt = time.time() - t0
        print(f"  {tag:>22}: {dt:6.2f}s   ({dt/len(frames)*1000:.0f} ms/帧)")
        return dt
    except Exception as e:
        print(f"  {tag:>22}: 失败 {str(e)[:110]}")
        return None

print("=== 特征提取 ===")
a = bench("threads=1", threads=1, gpu=False)
b = bench("threads=-1(默认)", threads=-1, gpu=False)
c = bench("threads=16", threads=16, gpu=False)
d = bench("use_gpu=True", threads=-1, gpu=True)
if a and b:
    print(f"\n  默认配置相对单线程加速 {a/b:.2f}x  （16 核理想值应接近 8~10x）")
if a and d:
    print(f"  GPU 相对单线程 {a/d:.2f}x" if d else "")
