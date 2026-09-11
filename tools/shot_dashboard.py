"""给看板截图：先加载，选一个运行与 checkpoint，看内嵌预览是否正常。"""
import sys
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

url, out = sys.argv[1], sys.argv[2]
opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--enable-unsafe-swiftshader")
opts.add_argument("--use-angle=swiftshader")
opts.add_argument("--window-size=1500,900")
opts.add_argument("--no-first-run")
d = webdriver.Chrome(options=opts)
try:
    d.get(url)
    time.sleep(6)
    print("note:", d.find_element("id", "note").text)
    # 选 cuda_long（最优运行）
    d.execute_script("""
      const items = [...document.querySelectorAll('#runs .item')];
      const t = items.find(e => e.textContent.includes('cuda_long'));
      if (t) t.click();
    """)
    time.sleep(3)
    cks = d.execute_script("return [...document.querySelectorAll('#cks .ck')].map(e=>e.textContent.trim())")
    print("checkpoints:", cks[:8])
    # 选最后一个（最强）
    d.execute_script("""
      const cks = [...document.querySelectorAll('#cks .ck')];
      if (cks.length) cks[cks.length-1].click();
    """)
    time.sleep(14)
    print("metrics:", d.find_element("id", "metrics").text.replace("\n", " | ")[:300])
    print("curveNote:", d.find_element("id", "curveNote").text)
    d.save_screenshot(out)
    print("screenshot ->", out)
finally:
    d.quit()
