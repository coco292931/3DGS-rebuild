"""实测顺序匹配阶段的线程效率。"""
import shutil
import time
from pathlib import Path

import pycolmap
from pycolmap import FeatureExtractionOptions, FeatureMatchingOptions

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "real4" / "frames"
TMP = ROOT / "output" / "_bench_match"
TMP.mkdir(parents=True, exist_ok=True)

frames = sorted(SRC.glob("*.png"))[:120]
print(f"用 {len(frames)} 张图测匹配\n")

def prep(tag):
    work = TMP / tag
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    (work / "frames").mkdir(parents=True)
    for f in frames:
        shutil.copy2(f, work / "frames" / f.name)
    db = work / "database.db"
    opts = FeatureExtractionOptions()
    opts.num_threads = -1
    try:
        mode = pycolmap.CameraMode.SINGLE
    except AttributeError:
        mode = "SINGLE"
    pycolmap.extract_features(str(db), str(work / "frames"), camera_mode=mode,
                              extraction_options=opts, device=pycolmap.Device.auto)
    return str(db)

def bench_match(tag, prep_tag, threads, gpu):
    db = prep(prep_tag)
    opts = FeatureMatchingOptions()
    opts.num_threads = threads
    opts.use_gpu = gpu
    t0 = time.time()
    try:
        pycolmap.match_sequential(db, matching_options=opts, device=pycolmap.Device.auto)
        dt = time.time() - t0
        print(f"  {tag:>20}: {dt:6.2f}s")
        return dt
    except Exception as e:
        print(f"  {tag:>20}: 失败 {str(e)[:100]}")
        return None

print("=== 顺序匹配 ===")
# 注意：匹配是增量的，同一个 db 重复跑会跳过已匹配的对，所以每次都要重新准备
a = bench_match("threads=1", "m1", 1, False)
b = bench_match("threads=-1(默认)", "m2", -1, False)
print()
if a and b:
    print(f"  默认相对单线程加速 {a/b:.2f}x")
