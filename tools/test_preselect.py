"""验证：从左侧数据集点「用这个数据集训练」应预选该数据集。"""
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
    time.sleep(5)

    # 点左侧「real3」数据集
    # .name 里内嵌了 .meta（帧数），textContent 会带上它，所以取第一个文本节点比较
    ok = d.execute_script("""
      const items = [...document.querySelectorAll('#datasets .item')];
      const t = items.find(e => {
        const n = e.querySelector('.name');
        return n && n.firstChild && n.firstChild.nodeValue.trim() === 'real3';
      });
      if (t) { t.click(); return true; }
      return false;
    """)
    print("点到 real3:", ok)
    time.sleep(1)

    # 选中后应出现「用这个数据集训练」按钮
    btn = d.execute_script("""
      const items = [...document.querySelectorAll('#datasets .item')];
      const t = items.find(e => e.classList.contains('sel'));
      if (!t) return null;
      const b = t.querySelector('button');
      if (!b) return null;
      const label = b.textContent;
      b.click();
      return label;
    """)
    print("点到的按钮:", btn)
    time.sleep(2)

    sel = Select(d.find_element("id", "f_dataset"))
    got = sel.first_selected_option.get_attribute("value")
    print("弹窗预选的数据集:", got, "  ->", "OK" if got == "real3" else "<<< 没有预选!")

    # 再确认跨刷新仍然保持
    time.sleep(9)
    got2 = Select(d.find_element("id", "f_dataset")).first_selected_option.get_attribute("value")
    print("9 秒后仍选中:", got2, "  ->", "OK" if got2 == "real3" else "<<< 被重置!")
finally:
    d.quit()
