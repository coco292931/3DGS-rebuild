"""验证看板能真的启动训练并跟踪进度。"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8770"


def post(path, obj):
    req = urllib.request.Request(BASE + path, method="POST",
                                 data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=30).read())
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "body": e.read().decode()[:200]}


def get(path):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=30).read())


print("数据集:", [f"{d['name']}({d['registered']})" for d in get("/api/datasets")["datasets"] if d["has_sfm"]])

# 用很小的步数启动一次真实训练，确认整条链路通
r = post("/api/train", {"dataset": "real4", "backend": "cuda", "iters": 200,
                        "downscale": 3, "out": "_ui_smoke"})
print("启动:", r)
job = r.get("id")
if not job:
    raise SystemExit("启动失败")

for i in range(8):
    time.sleep(4)
    jobs = get("/api/datasets")["jobs"]
    j = next((x for x in jobs if x["id"] == job), None)
    if not j:
        break
    log = get(f"/api/joblog?name={job}")["log"]
    tail = [l for l in log.splitlines() if ">>>" in l or "loss=" in l][-1:]
    print(f"  [{i}] alive={j['alive']} {tail}")
    if not j["alive"]:
        print("  任务结束 rc =", j["returncode"])
        break
print("\n最终日志尾部:")
print("\n".join(get(f"/api/joblog?name={job}")["log"].splitlines()[-6:]))
