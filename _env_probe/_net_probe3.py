import urllib.request, json, time
res={}
for u in ["https://codeload.github.com/graphdeco-inria/gaussian-splatting/tar.gz/refs/heads/main",
          "https://api.github.com/repos/graphdeco-inria/gaussian-splatting",
          "https://huggingface.co/api/models?limit=1"]:
    t0=time.time()
    try:
        req=urllib.request.Request(u, headers={"User-Agent":"Mozilla/5.0"})
        r=urllib.request.urlopen(req, timeout=15)
        res[u]="OK %d %.1fs len=%s" % (r.status, time.time()-t0, r.headers.get("Content-Length"))
    except Exception as e:
        res[u]="FAIL %.1fs %s" % (time.time()-t0, repr(e)[:100])
open(r"C:\Users\tonyp\Downloads\_net3.json","w").write(json.dumps(res,indent=1))
print("done")
