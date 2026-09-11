"""验证异步导入：立刻返回、进度可查、中途不谎报帧数。"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8770"


def post(p, o):
    req = urllib.request.Request(BASE + p, method="POST", data=json.dumps(o).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def get(p):
    return json.loads(urllib.request.urlopen(BASE + p, timeout=30).read())


t0 = time.time()
r = post("/api/import", {"path": r"C:\Users\tonyp\Downloads\3d-rebuild\VID_20260911_124622.mp4",
                         "name": "figurine", "fps": 6, "scale": 1280})
print(f"启动导入耗时 {time.time()-t0:.2f} 秒 -> {r}  (应当立刻返回，不阻塞)")
job = r["id"]

for i in range(28):
    time.sleep(15)
    log = get("/api/joblog?name=" + job)["log"]
    ds = get("/api/datasets")
    me = next((x for x in ds["datasets"] if x["name"] == "figurine"), None)
    jobs = [j for j in ds["jobs"] if j["id"] == job]
    alive = jobs[0]["alive"] if jobs else None
    lines = [l for l in log.splitlines() if l.strip()]
    print(f"  [{i:2d}] importing={me['importing'] if me else '?':<5} "
          f"registered={me['registered'] if me else '?':<5} 运行中={alive}")
    print(f"       {lines[-1] if lines else '(等待输出)'}")
    if alive is False:
        print("  结束")
        break
print("\n最终状态：", {k: v for k, v in (me or {}).items() if k in ("name", "images", "has_sfm", "registered", "importing")})
