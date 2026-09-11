"""视频导入：抽帧 + SfM，产出一个可直接训练的数据集。

刻意做成独立脚本而不是在看板进程里开线程——线程随看板进程一起死，
看板一重启/被关掉，正在跑的导入就被拦腰砍断，留下半截 database.db。

用法：
    python tools/import_video.py --video <path> --name <dataset> [--fps 6] [--scale 1280]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FFMPEG_CANDIDATES = [
    Path(r"C:\Users\tonyp\Downloads\FFmpeg\bin\ffmpeg.exe"),
    Path("ffmpeg"),
]


def find_ffmpeg() -> str:
    for c in FFMPEG_CANDIDATES:
        if c.exists() or not c.is_absolute():
            return str(c)
    return "ffmpeg"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--scale", type=int, default=1280)
    args = ap.parse_args()

    src = Path(args.video)
    if not src.exists():
        print(f"视频不存在：{src}")
        return 2
    name = re.sub(r"[^0-9A-Za-z_]+", "_", args.name)[:40] or "video"
    ds = ROOT / "data" / name
    frames = ds / "frames"
    marker = ds / ".importing"

    ds.mkdir(parents=True, exist_ok=True)
    marker.write_text("1", encoding="utf-8")
    frames.mkdir(parents=True, exist_ok=True)
    for old in frames.glob("*.png"):
        old.unlink()

    def say(m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    def run(cmd, label):
        say(f"{label} …")
        p = subprocess.run([str(c) for c in cmd], cwd=str(ROOT),
                           capture_output=True, text=True)
        tail = ((p.stdout or "") + (p.stderr or "")).strip()[-500:]
        if tail:
            print("    " + tail.replace("\n", "\n    "), flush=True)
        if p.returncode != 0:
            raise RuntimeError(f"{label} 失败（返回码 {p.returncode}）")

    try:
        say(f"开始导入 {src.name} -> {name}")
        # fps 必须够密：手持转半秒能转过二十几度，帧间视差过大会直接匹配失败
        run([find_ffmpeg(), "-y", "-v", "error", "-i", src,
             "-vf", f"fps={args.fps},scale={args.scale}:-1", "-q:v", "2",
             frames / "frame_%04d.png"], "抽帧")
        n = len(list(frames.glob("*.png")))
        say(f"抽出 {n} 帧")
        if n < 12:
            raise RuntimeError(f"只抽出 {n} 帧，太少，无法重建")

        for stage, label in (("features", "特征提取"), ("match", "顺序匹配"), ("map", "增量式重建")):
            cmd = [sys.executable, "-u", str(ROOT / "tools" / "run_sfm.py"),
                   f"data/{name}", "--stage", stage]
            if stage == "match":
                cmd.append("--sequential")
            run(cmd, label)

        n_reg = 0
        try:
            import pycolmap
            from src.colmap_dataset import ColmapDataset
            n_reg = pycolmap.Reconstruction(str(ColmapDataset._find_model(ds))).num_reg_images()
        except Exception as e:
            say(f"读回模型失败：{str(e)[:150]}")
        say(f"完成：{n} 帧抽出，{n_reg} 帧注册")
        return 0
    except Exception as e:
        say(f"失败：{e}")
        return 1
    finally:
        marker.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
