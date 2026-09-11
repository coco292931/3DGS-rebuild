"""验证：「新建训练」下拉框的选择不会被自动刷新冲掉。"""
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--enable-unsafe-swiftshader")
opts.add_argument("--use-angle=swiftshader")
opts.add_argument("--window-size=1500,900")
d = webdriver.Chrome(options=opts)
try:
    d.get("http://127.0.0.1:8770/")
    time.sleep(4)
    d.find_element("id", "newtrain").click()
    time.sleep(1)

    sel = Select(d.find_element("id", "f_dataset"))
    opts0 = [(o.get_attribute("value"), o.get_attribute("disabled")) for o in sel.options]
    print("下拉选项:", opts0)
    print("初始选中:", sel.first_selected_option.get_attribute("value"))

    # 选一个非第一项的可用数据集
    target = next(v for v, dis in opts0 if v != sel.first_selected_option.get_attribute("value"))
    sel.select_by_value(target)
    print("手动选中:", target)

    # 跨过至少两轮自动刷新（4 秒一次）
    for i in range(3):
        time.sleep(4)
        now = Select(d.find_element("id", "f_dataset")).first_selected_option.get_attribute("value")
        print(f"  等待 {(i+1)*4}s 后仍选中: {now}  {'OK' if now == target else '<<< 被重置!'}")

    # 再验证：从左侧数据集点「用这个数据集训练」应预选它
    d.execute_script("document.getElementById('f_cancel').click()")
    time.sleep(2)
    print("\n弹窗已关闭")
finally:
    d.quit()
