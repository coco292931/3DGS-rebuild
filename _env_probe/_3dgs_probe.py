import sys, os, json, time, io
out = {}
out["python"] = sys.version
try:
    import torch
    out["torch"] = torch.__version__
    out["torch_cuda"] = torch.version.cuda
    out["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        out["gpu"] = p.name
        out["vram_GB"] = round(p.total_memory/1e9, 2)
        out["sm"] = str(p.major)+"."+str(p.minor)
        out["arch_list"] = torch.cuda.get_arch_list()
        a = torch.randn(2000, 2000, device="cuda")
        t0 = time.time(); b = a @ a; torch.cuda.synchronize()
        out["matmul_ms"] = round((time.time()-t0)*1000, 1)
except Exception as e:
    out["torch_error"] = repr(e)
for m in ["cv2","open3d","numpy","scipy","matplotlib","imageio","plyfile","trimesh","PIL"]:
    try:
        mod = __import__(m); out[m] = getattr(mod, "__version__", "ok")
    except Exception:
        out[m] = "MISSING"
# network test with python's own stack
import urllib.request, ssl
for url in ["https://pypi.org/simple/", "https://hf-mirror.com", "https://github.com"]:
    try:
        t0=time.time()
        r = urllib.request.urlopen(url, timeout=10)
        out["net:"+url] = "OK %d in %.1fs" % (r.status, time.time()-t0)
    except Exception as e:
        out["net:"+url] = "FAIL " + repr(e)[:120]
open(r"C:\Users\tonyp\Downloads\_probe_out.json","w",encoding="utf-8").write(json.dumps(out, indent=1, ensure_ascii=False))
print("written")
