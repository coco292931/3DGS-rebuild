"""粗查：函数体内用到某个模块、但该模块既没在文件顶部导入、也没在函数内导入。

只针对已知的标准库模块名（否则 foo.bar() 里的局部变量会被误报成模块）。
专门用来抓「把实现从 A 换成 B 却忘了改 import」这类错误——
上一次就是这样让 /api/import 报了 name 'subprocess' is not defined。
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 只查这些名字：它们几乎只会以「模块」身份出现在 X.attr 里
MODULES = {
    "subprocess", "threading", "json", "os", "sys", "re", "time", "math",
    "struct", "shutil", "glob", "random", "tempfile", "urllib", "pathlib",
    "mimetypes", "argparse", "traceback", "warnings", "collections",
    "np", "torch", "Image", "pycolmap", "ffmpeg", "cv2",
}

FILES = ["tools/dashboard.py", "tools/import_video.py", "tools/run_sfm.py",
         "src/train.py", "src/rasterizer.py", "src/rasterizer_cuda.py",
         "src/gaussian_model.py", "src/colmap_dataset.py", "src/dataset.py",
         "src/camera.py", "src/sh.py", "src/metrics.py"]

bad = 0
for rel in FILES:
    p = ROOT / rel
    if not p.exists():
        continue
    tree = ast.parse(p.read_text(encoding="utf-8"))

    top = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                top.add(a.asname or a.name)

    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        local = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Import):
                for a in node.names:
                    local.add((a.asname or a.name).split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    local.add(a.asname or a.name)
        used = {n.value.id for n in ast.walk(fn)
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
        missing = sorted(m for m in used & MODULES if m not in top and m not in local)
        if missing:
            bad += 1
            print(f"  {rel}:{fn.lineno}  {fn.name}()  缺少导入: {missing}")

print("检查完成:", f"发现 {bad} 处问题" if bad else "未发现明显遗漏")
sys.exit(1 if bad else 0)
