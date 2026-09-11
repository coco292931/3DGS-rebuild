import subprocess, json, shutil, os, sys
res = {"which_ninja": shutil.which("ninja")}
for mode, kw in [("capture_output", dict(capture_output=True)), ("pipe", dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE)), ("inherit", dict())]:
    try:
        p = subprocess.run(["ninja", "--version"], **kw)
        out = ""
        if kw:
            out = str(getattr(p, "stdout", b""))[:60]
        res[mode] = "OK rc=%s out=%s" % (p.returncode, out)
    except Exception as e:
        res[mode] = "FAIL %s: %s" % (type(e).__name__, str(e)[:150])
try:
    from torch.utils.cpp_extension import _is_ninja_available
    res["torch_is_ninja_available"] = _is_ninja_available()
except Exception as e:
    res["torch_is_ninja_available"] = "ERR " + repr(e)[:120]
nvcc = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe"
try:
    p = subprocess.run([nvcc, "--version"], capture_output=True)
    res["nvcc_capture"] = "OK rc=%s %s" % (p.returncode, p.stdout.decode(errors="replace").strip().splitlines()[-1][:80] if p.stdout else "")
except Exception as e:
    res["nvcc_capture"] = "FAIL %s: %s" % (type(e).__name__, str(e)[:150])
open(r"C:\Users\tonyp\Downloads\3dgs\_env_probe\sandbox_stdio_result.json", "w", encoding="utf-8").write(json.dumps(res, indent=1, ensure_ascii=False))
print("done")
