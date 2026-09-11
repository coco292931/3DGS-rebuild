import json, urllib.request, time
urls = [
 "https://cdn.jsdelivr.net/gh/graphdeco-inria/gaussian-splatting@main/README.md",
 "https://fastly.jsdelivr.net/gh/graphdeco-inria/gaussian-splatting@main/README.md",
 "https://gcore.jsdelivr.net/gh/graphdeco-inria/gaussian-splatting@main/README.md",
 "https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/README.md",
 "https://gh-proxy.com/https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/README.md",
 "https://ghfast.top/https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/README.md",
 "https://arxiv.org/abs/2308.04079",
 "https://hf-mirror.com",
 "https://gitee.com",
 "https://pypi.org/simple/plyfile/",
]
res = {}
for u in urls:
    t0 = time.time()
    try:
        req = urllib.request.Request(u, headers={"User-Agent":"Mozilla/5.0"})
        r = urllib.request.urlopen(req, timeout=12)
        body = r.read(400)
        res[u] = "OK %d len=%s %.1fs head=%r" % (r.status, r.headers.get("Content-Length"), time.time()-t0, body[:80])
    except Exception as e:
        res[u] = "FAIL %.1fs %s" % (time.time()-t0, repr(e)[:90])
open(r"C:\Users\tonyp\Downloads\_net2.json","w",encoding="utf-8").write(json.dumps(res, indent=1, ensure_ascii=False))
print("done")
