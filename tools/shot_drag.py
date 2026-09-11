"""用真实的鼠标拖拽事件驱动查看器，证明「视角可移动」是真的交互链路，
不是改 JS 变量后重渲染的静态图。

走的是浏览器真实输入管线：mousedown -> mousemove -> mouseup。
"""
import json
import sys
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By

url, prefix = sys.argv[1], sys.argv[2]
prefix = Path(prefix)

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--enable-unsafe-swiftshader")
opts.add_argument("--use-angle=swiftshader")
opts.add_argument("--window-size=760,540")
opts.add_argument("--no-first-run")

d = webdriver.Chrome(options=opts)
shots = []
try:
    d.get(url)
    time.sleep(10)
    canvas = d.find_element(By.ID, "gl")

    def shot(tag):
        p = prefix.parent / f"{prefix.name}_{tag}.png"
        d.save_screenshot(str(p))
        shots.append(p)
        yaw = d.execute_script("return cam.yaw")
        pitch = d.execute_script("return cam.pitch")
        dist = d.execute_script("return cam.dist")
        print(f"{tag:>12}  yaw={yaw:+.3f}  pitch={pitch:+.3f}  dist={dist:.2f}")
        return p

    shot("1_initial")

    # 真实鼠标拖拽：按住左键横向拖动
    ActionChains(d).move_to_element(canvas).click_and_hold().move_by_offset(150, 0).release().perform()
    time.sleep(3.0)
    shot("2_drag_right")

    ActionChains(d).move_to_element(canvas).click_and_hold().move_by_offset(-330, 40).release().perform()
    time.sleep(3.0)
    shot("3_drag_left")

    ActionChains(d).move_to_element(canvas).click_and_hold().move_by_offset(120, -120).release().perform()
    time.sleep(3.0)
    shot("4_drag_up")

    # 真实滚轮事件
    d.execute_script("""
      const cv = document.getElementById('gl');
      cv.dispatchEvent(new WheelEvent('wheel', {deltaY: -600, bubbles: true, cancelable: true}));
    """)
    time.sleep(3.0)
    shot("5_zoom_in")
finally:
    d.quit()

from PIL import Image, ImageDraw
imgs = [Image.open(p).convert("RGB") for p in shots]
w, h = imgs[0].size
cols, rows = 2, (len(imgs) + 1) // 2
grid = Image.new("RGB", (w * cols, h * rows), (16, 20, 26))
for i, im in enumerate(imgs):
    grid.paste(im, ((i % cols) * w, (i // cols) * h))
grid.save(prefix.parent / f"{prefix.name}_grid.png")
print("grid ->", prefix.parent / f"{prefix.name}_grid.png")
