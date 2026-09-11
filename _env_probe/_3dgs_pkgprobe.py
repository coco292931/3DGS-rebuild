import json, urllib.request
res = {}
for name in ["pycolmap", "gsplat", "plyfile", "diff-gaussian-rasterization", "torch", "nvidia-cuda-nvcc-cu13"]:
    try:
        d = json.load(urllib.request.urlopen("https://pypi.org/pypi/%s/json" % name, timeout=15))
        vs = sorted(d["releases"].keys())[-6:]
        files = []
        latest = d["info"]["version"]
        for f in d["releases"][latest]:
            files.append(f["filename"])
        res[name] = {"latest": latest, "recent": vs, "latest_files": files[:8]}
    except Exception as e:
        res[name] = "ERR " + repr(e)[:100]
open(r"C:\Users\tonyp\Downloads\_pkg_probe.json","w",encoding="utf-8").write(json.dumps(res, indent=1, ensure_ascii=False))
print("ok")
