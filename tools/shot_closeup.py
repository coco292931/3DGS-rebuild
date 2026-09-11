"""把相机推到很近的距离，检验近处淡出与屏幕半径剔除是否有效。

用户把滚轮推近时，贴脸的高斯会糊满屏幕；这里逐档逼近，看画面是否仍然可用。
"""
import sys
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

url, prefix = sys.argv[1], sys.argv[2]
factors = [1.0, 0.45, 0.22, 0.10, 0.05]

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
    r = d.execute_script("return sceneRadius")
    print("sceneRadius =", round(float(r), 4))
    for f in factors:
        d.execute_script(f"cam.dist = sceneRadius * {f}; return 1;")
        time.sleep(2.2)
        p = Path(f"{prefix}_{f}.png")
        d.save_screenshot(str(p))
        shots.append((p, f))
        print(f"dist = sceneRadius*{f:<5} -> {p.name}")
finally:
    d.quit()

from PIL import Image, ImageDraw
ims = [Image.open(p).convert("RGB") for p, _ in shots]
w, h = ims[0].size
grid = Image.new("RGB", (w * len(ims), h), (16, 20, 26))
for i, im in enumerate(ims):
    grid.paste(im, (i * w, 0))
    ImageDraw.Draw(grid).text((i * w + 8, 6), f"dist = {shots[i][1]}x scene", fill=(255, 220, 90))
grid.save(f"{prefix}_grid.png")
print("grid ->", f"{prefix}_grid.png")
