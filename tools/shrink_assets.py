"""压缩仓库里的截图：PNG -> JPEG。

对比图、截图这类内容用 JPEG 足够，PNG 无损格式在这里纯属浪费体积。
"""
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DIRS = ["output/deliverable", "output/deliverable_real",
        "output/deliverable_real4", "output/deliverable_figurine"]
QUALITY = 86

total_before = total_after = 0
for d in DIRS:
    p = ROOT / d
    if not p.exists():
        continue
    for png in sorted(p.glob("*.png")):
        before = png.stat().st_size
        im = Image.open(png).convert("RGB")
        jpg = png.with_suffix(".jpg")
        im.save(jpg, "JPEG", quality=QUALITY, optimize=True, progressive=True)
        after = jpg.stat().st_size
        png.unlink()
        total_before += before
        total_after += after
        print(f"  {d}/{png.name}: {before//1024} KB -> {jpg.name} {after//1024} KB"
              f"  ({after/before*100:.0f}%)")

print(f"\n合计 {total_before/1e6:.2f} MB -> {total_after/1e6:.2f} MB"
      f"（压掉 {100-total_after/total_before*100:.0f}%）")
