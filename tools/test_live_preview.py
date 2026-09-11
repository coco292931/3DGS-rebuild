"""验证训练中的实时预览：起一个短训练，轮询 preview/ 目录看新图是否不断出现。"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8770"


def post(p, o):
    req = urllib.request.Request(BASE + p, method="POST", data=json.dumps(o).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def get(p):
    return json.loads(urllib.request.urlopen(BASE + p, timeout=25).read())


r = post("/api/train", {"dataset": "ui_log", "backend": "cuda", "iters": 800, "downscale": 3,
                        "ckpt_interval": 200, "preview_interval": 100, "out": "_live_demo"})
print("启动:", r)
job = r["id"]
seen = 0
for i in range(12):
    time.sleep(5)
    pv = get("/api/previews?name=_live_demo")
    jobs = get("/api/datasets")["jobs"]
    j = next((x for x in jobs if x["id"] == job), None)
    n = len(pv["files"])
    mark = "  <-- 有新图" if n > seen else ""
    seen = n
    print(f"  [{i}] 预览 {n} 张, 最新 {pv['latest']}, 运行中={j['alive'] if j else None}{mark}")
    if j and not j["alive"]:
        print("  任务结束 rc =", j["returncode"])
        break
print("最终预览文件:", get("/api/previews?name=_live_demo")["files"])
