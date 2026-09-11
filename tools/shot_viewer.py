"""用无头 Chrome 打开查看器并截图，验证它真的能渲染（而不是靠嘴说）。

用法： python tools/shot_viewer.py <url> <out.png> [等待秒数]
"""
import sys, time, json

from selenium import webdriver
from selenium.webdriver.chrome.options import Options


def main():
    url, out = sys.argv[1], sys.argv[2]
    wait = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0

    import os
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--enable-unsafe-swiftshader")   # 无 GPU 环境下用软件 WebGL
    angle = os.environ.get("CHROME_ANGLE", "swiftshader")
    if angle != "none":
        opts.add_argument(f"--use-angle={angle}")
    opts.add_argument("--window-size=900,620")
    opts.add_argument("--no-first-run")
    opts.add_argument("--no-default-browser-check")
    opts.add_argument("--disable-extensions")
    opts.set_capability("goog:loggingPrefs", {"browser": "ALL"})

    driver = webdriver.Chrome(options=opts)
    try:
        driver.get(url)
        time.sleep(wait)
        for eid in ("hud", "log"):
            try:
                print(f"{eid}:", driver.find_element("id", eid).text.replace("\n", " | "))
            except Exception:
                pass
        logs = driver.get_log("browser")
        for e in logs:
            print(f"[console.{e['level']}] {e['message'][:300]}")
        driver.save_screenshot(out)
        print("screenshot ->", out)
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
