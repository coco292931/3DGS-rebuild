import os, sys, json, time, traceback, subprocess
os.environ.setdefault("TORCH_EXTENSIONS_DIR", r"C:\Users\tonyp\Downloads\3dgs\_build\ext")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
# 优先使用最新安装的 CUDA toolkit（v13.4），覆盖系统环境里的 v12.8
import glob as _glob, re as _re
_c = _glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*")
def _v(p):
    m = _re.search(r"v(\d+)\.(\d+)", os.path.basename(p))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
_c.sort(key=_v)
if _c:
    os.environ["CUDA_HOME"] = _c[-1]
    os.environ["CUDA_PATH"] = _c[-1]
res = {"cuda_home_env": os.environ.get("CUDA_HOME"), "arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST")}
# make the freshly installed ninja visible to torch's build machinery
import glob, shutil
extra = []
try:
    import site, ninja as _nj
    pkg = os.path.dirname(_nj.__file__)
    for cand in glob.glob(os.path.join(pkg, "**", "ninja.exe"), recursive=True):
        extra.append(os.path.dirname(cand))
    for sp in site.getusersitepackages() and [site.getusersitepackages()] or []:
        extra.append(os.path.join(os.path.dirname(sp), "Scripts"))
except Exception as e:
    res["ninja_import_error"] = repr(e)
for d in extra:
    if os.path.isdir(d) and d not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
res["ninja_paths"] = extra
res["ninja_which"] = shutil.which("ninja")
res["cuda_toolkits"] = _c
try:
    _nv = os.path.join(os.environ.get("CUDA_HOME", ""), "bin", "nvcc.exe")
    _p = subprocess.run([_nv, "--version"], capture_output=True, text=True)
    res["nvcc"] = [l for l in _p.stdout.splitlines() if "release" in l][0].strip()
except Exception as _e:
    res["nvcc"] = "ERR " + repr(_e)[:120]
import torch
res["torch"] = torch.__version__
res["torch_cuda"] = torch.version.cuda
res["cuda_available"] = torch.cuda.is_available()
res["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
from torch.utils.cpp_extension import load_inline, CUDA_HOME
res["CUDA_HOME_resolved"] = str(CUDA_HOME)
t0 = time.time()
cuda_src = r'''
#include <torch/extension.h>
__global__ void bump_kernel(float* p) { p[threadIdx.x] += 1.0f; }
torch::Tensor bump(torch::Tensor x) {
    auto y = x.clone();
    bump_kernel<<<1, 8>>>(y.data_ptr<float>());
    return y;
}
'''
cpp_src = "torch::Tensor bump(torch::Tensor x);"
try:
    mod = load_inline(name="dsh_cuda_probe_cu134b", cpp_sources=cpp_src,
                      cuda_sources=cuda_src, functions=["bump"], verbose=False,
                      extra_cuda_cflags=["-O1", "-Xcompiler", "/Zc:preprocessor"],
                      extra_cflags=["/Zc:preprocessor"])
    res["compile"] = "OK"
    res["compile_sec"] = round(time.time() - t0, 1)
    y = mod.bump(torch.zeros(8, device="cuda"))
    torch.cuda.synchronize()
    res["kernel_out"] = y.tolist()
    res["verdict"] = "TOOLCHAIN_WORKS"
except Exception as e:
    res["compile"] = "FAIL"
    res["compile_sec"] = round(time.time() - t0, 1)
    res["error_type"] = type(e).__name__
    res["error_head"] = str(e)[:400]
    open(r"C:\Users\tonyp\Downloads\3dgs\_env_probe\cuda_toolchain_error.txt", "w", encoding="utf-8").write(str(e))
    res["verdict"] = "TOOLCHAIN_BROKEN"
open(r"C:\Users\tonyp\Downloads\3dgs\_env_probe\cuda_toolchain_result.json", "w", encoding="utf-8").write(json.dumps(res, indent=1, ensure_ascii=False))
print("verdict:", res["verdict"], "| compile:", res.get("compile"), "|", res.get("compile_sec"), "s")
