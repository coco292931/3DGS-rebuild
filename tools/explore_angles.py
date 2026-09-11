"""扫一批视角，找出真实场景重建里最可看的角度。"""
import json
import sys
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

url, prefix = sys.argv[1], sys.argv[2]
angles = json.loads(sys.argv[3])

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--enable-unsafe-swiftshader")
opts.add_argument("--use-angle=swiftshader")
opts.add_argument("--window-size=700,500")
opts.add_argument("--no-first-run")
d = webdriver.Chrome(options=opts)
shots = []
try:
    d.get(url)
    time.sleep(11)
    d.execute_script("document.getElementById('hud').style.display='none';")
    info = d.execute_script("return JSON.stringify({dist: cam.dist, target: cam.target})")
    print("auto framing:", info)
    for i, (yaw, pitch) in enumerate(angles):
        d.execute_script(f"cam.yaw = {yaw}; cam.pitch = {pitch}; return 1;")
        time.sleep(2.2)
        p = Path(f"{prefix}_{i}.png")
        d.save_screenshot(str(p))
        shots.append((p, yaw, pitch))
finally:
    d.quit()

from PIL import Image, ImageDraw
ims = [Image.open(p).convert("RGB") for p, _, _ in shots]
w, h = ims[0].size
cols = 4
rows = (len(ims) + cols - 1) // cols
grid = Image.new("RGB", (w * cols, h * rows), (16, 20, 26))
for i, im in enumerate(ims):
    x, y = (i % cols) * w, (i // cols) * h
    grid.paste(im, (x, y))
    yaw, pitch = shots[i][1], shots[i][2]
    ImageDraw.Draw(grid).text((x + 8, y + 6), f"yaw={yaw} pitch={pitch}", fill=(255, 220, 90))
    ImageDraw.Draw(grid).rectangle([x, y, x + w - 1, y + h - 1], outline=(80, 90, 110))
grid.save(f"{prefix}_grid.png")
print("grid ->", f"{prefix}_grid.png")
