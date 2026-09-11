"""训练监控看板的后端。

一个零依赖的 HTTP 服务（只用标准库），给展示用的前端页面提供：
    GET /                    看板页面
    GET /api/runs            列出所有训练运行及其状态
    GET /api/run/<name>      某个运行的详情（checkpoint 列表、日志、指标）
    GET /model/<name>/<file> PLY 模型文件
    GET /viewer.html         单场景查看器（看板用它做内嵌预览）

用法：
    python tools/dashboard.py                # 默认扫 output/，端口 8770
    python tools/dashboard.py --port 8770 --root output
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, parse_qs

from src.colmap_dataset import ColmapDataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # 让 tools/ 下也能 import src.*
    sys.path.insert(0, str(ROOT))
OUTPUT_ROOT = ROOT / "output"
VIEWER = ROOT / "viewer" / "viewer.html"

# 训练正在写入时，文件 mtime 在这么多秒内视为「活跃」
ACTIVE_WINDOW = 90


def _ckpt_iter(p: Path) -> int:
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else -1


def scan_runs() -> list[dict]:
    """扫描 output/ 下的训练运行。"""
    runs = []
    if not OUTPUT_ROOT.exists():
        return runs
    for d in sorted(OUTPUT_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir() or d.name.startswith("_") or d.name.startswith("deliverable"):
            continue
        ckpts = sorted((d / "ckpt").glob("*.ply"), key=_ckpt_iter) if (d / "ckpt").exists() else []
        previews = sorted((d / "preview").glob("*.png"), key=_ckpt_iter) if (d / "preview").exists() else []
        final_ply = d / "point_cloud.ply"
        summary = d / "summary.json"
        log = d / "train_log.jsonl"

        latest_mtime = max([p.stat().st_mtime for p in [*ckpts, *previews, log, final_ply, summary]
                            if p.exists()] or [d.stat().st_mtime])
        info = {
            "name": d.name,
            "path": str(d),
            "active": (time.time() - latest_mtime) < ACTIVE_WINDOW,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(latest_mtime)),
            "checkpoints": [{"name": p.name, "iter": _ckpt_iter(p), "size_mb": round(p.stat().st_size / 1e6, 1)}
                            for p in ckpts],
            "previews": [p.name for p in previews],
            "has_final": final_ply.exists(),
            "summary": None,
            "log_lines": 0,
        }
        if summary.exists():
            try:
                info["summary"] = json.loads(summary.read_text(encoding="utf-8"))
            except Exception:
                pass
        if log.exists():
            try:
                info["log_lines"] = sum(1 for _ in log.open(encoding="utf-8"))
            except Exception:
                pass
        runs.append(info)
    return runs


def read_log(name: str, tail: int = 5000) -> list[dict]:
    p = OUTPUT_ROOT / name / "train_log.jsonl"
    if not p.exists():
        return []
    out = []
    try:
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass
    return out[-tail:]




# ---------------------------------------------------------------------------
# 数据集与训练任务
# ---------------------------------------------------------------------------
JOBS: dict[str, dict] = {}
JOBS_DIR = ROOT / "output" / "_jobs"


def scan_datasets() -> list[dict]:
    """扫描 data/ 下的可用数据集，标注是否已经做过 SfM。"""
    data_root = ROOT / "data"
    out = []
    if not data_root.exists():
        return out
    for d in sorted(data_root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        frames = d / "frames"
        imgs = []
        for sub in ("frames", "images"):
            p = d / sub
            if p.exists():
                imgs = sorted([*p.glob("*.png"), *p.glob("*.jpg")])
                break
        has_sfm = (d / "sparse").exists()
        n_view = 0
        if has_sfm:
            try:
                import pycolmap
                best = ColmapDataset._find_model(d)
                n_view = pycolmap.Reconstruction(str(best)).num_reg_images()
            except Exception:
                n_view = -1
        out.append({
            "name": d.name,
            "images": len(imgs),
            "has_sfm": has_sfm,
            "registered": n_view,
            "kind": "colmap" if has_sfm else ("images" if imgs else "empty"),
        })
    return out


def list_jobs() -> list[dict]:
    jobs = []
    for name, j in JOBS.items():
        p = j["proc"]
        alive = p.poll() is None
        jobs.append({
            "id": name,
            "kind": j["kind"],
            "cmd": j["cmd_short"],
            "alive": alive,
            "returncode": None if alive else p.returncode,
            "started": j["started"],
        })
    return jobs


def read_job_log(name: str, tail: int = 400) -> str:
    p = JOBS_DIR / f"{name}.log"
    if not p.exists():
        return ""
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-tail:])
    except Exception:
        return ""


def start_job(kind: str, args: dict) -> dict:
    """在后台起一个训练或 SfM 子进程，输出重定向到文件。"""
    import subprocess

    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{kind}_{time.strftime('%m%d_%H%M%S')}"

    if kind == "train":
        ds = str(args["dataset"])
        backend = args.get("backend", "cuda")
        out_name = args.get("out") or f"ui_{ds}"
        cmd = [
            sys.executable, "-u", "-m", "src.train",
            "--data", f"data/{ds}",
            "--backends" if False else "--backend", backend,
            "--iters", str(args.get("iters", 30000)),
            "--downscale", str(args.get("downscale", 2)),
            "--out", f"output/{out_name}",
            "--init-scale-factor", str(args.get("init_scale", 1.0)),
            "--max-gaussians", str(args.get("max_gaussians", 60000)),
            "--densify-until", str(args.get("densify_until", 15000)),
            "--opacity-reset-interval", str(args.get("opacity_reset", 3000)),
            "--ckpt-interval", "5000", "--log-interval", "500", "--eval-interval", "5000",
        ]
        if backend == "torch":
            cmd += ["--K", str(args.get("K", 64)), "--max-fragments", "16000000"]
        cmd_short = f"训练 {ds} · {backend} · {args.get('iters', 30000)} 步"
    elif kind == "sfm":
        ds = str(args["dataset"])
        cmd = [sys.executable, "-u", "tools/run_sfm.py", f"data/{ds}", "--stage", "all", "--sequential"]
        cmd_short = f"SfM {ds}"
    else:
        raise ValueError(f"未知任务类型 {kind}")

    log_path = JOBS_DIR / f"{name}.log"
    fh = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    JOBS[name] = {"proc": proc, "kind": kind, "cmd_short": cmd_short,
                  "started": time.strftime("%H:%M:%S"), "log": str(log_path)}
    return {"id": name, "cmd": cmd_short}

class Handler(BaseHTTPRequestHandler):
    server_version = "3dgs-dashboard/1.0"

    def log_message(self, fmt, *args):      # 静音访问日志
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            pass

    def do_POST(self):
        path = unquote(self.path.split("?", 1)[0])
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._json({"error": f"bad body: {e}"}, 400)

        if path == "/api/train":
            if not body.get("dataset"):
                return self._json({"error": "缺少 dataset"}, 400)
            try:
                return self._json(start_job("train", body))
            except Exception as e:
                return self._json({"error": str(e)}, 500)

        if path == "/api/sfm":
            if not body.get("dataset"):
                return self._json({"error": "缺少 dataset"}, 400)
            try:
                return self._json(start_job("sfm", body))
            except Exception as e:
                return self._json({"error": str(e)}, 500)

        if path == "/api/kill":
            name = body.get("id", "")
            j = JOBS.get(name)
            if not j:
                return self._json({"error": "没有这个任务"}, 404)
            try:
                j["proc"].terminate()
            except Exception:
                pass
            return self._json({"ok": True})

        return self._json({"error": "not found"}, 404)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        path = unquote(self.path.split("?", 1)[0])

        if path in ("/", "/index.html"):
            html = (Path(__file__).with_name("dashboard.html")).read_bytes()
            return self._send(200, html, "text/html; charset=utf-8")

        if path == "/viewer.html":
            return self._send(200, VIEWER.read_bytes(), "text/html; charset=utf-8")

        if path == "/api/runs":
            return self._json({"runs": scan_runs(), "time": time.strftime("%H:%M:%S")})

        if path.startswith("/api/run/"):
            name = path[len("/api/run/"):]
            base = (OUTPUT_ROOT / name).resolve()
            if not str(base).startswith(str(OUTPUT_ROOT.resolve())) or not base.is_dir():
                return self._json({"error": "not found"}, 404)
            return self._json({
                "name": name,
                "log": read_log(name),
                "summary": json.loads((base / "summary.json").read_text(encoding="utf-8"))
                           if (base / "summary.json").exists() else None,
            })


        # ---- 数据集与训练任务 ----
        if path == "/api/datasets":
            return self._json({"datasets": scan_datasets(), "jobs": list_jobs()})

        if path == "/api/joblog":
            q = self.path.split("?", 1)[1] if "?" in self.path else ""
            name = dict(p.split("=", 1) for p in q.split("&") if "=" in p).get("name", "")
            return self._json({"log": read_job_log(name)})

        if path.startswith("/model/"):
            rest = path[len("/model/"):]
            if "/" not in rest:
                return self._json({"error": "bad path"}, 400)
            name, fname = rest.split("/", 1)
            base = (OUTPUT_ROOT / name).resolve()
            target = (base / fname).resolve()
            if not str(target).startswith(str(base)) or not target.is_file():
                return self._json({"error": "not found"}, 404)
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            return self._send(200, target.read_bytes(), ctype)

        if path.startswith("/preview/"):
            rest = path[len("/preview/"):]
            if "/" not in rest:
                return self._json({"error": "bad path"}, 400)
            name, fname = rest.split("/", 1)
            target = (OUTPUT_ROOT / name / "preview" / fname).resolve()
            base = (OUTPUT_ROOT / name).resolve()
            if not str(target).startswith(str(base)) or not target.is_file():
                return self._json({"error": "not found"}, 404)
            return self._send(200, target.read_bytes(), "image/png")

        return self._json({"error": "not found"}, 404)


def main() -> int:
    global OUTPUT_ROOT
    ap = argparse.ArgumentParser(description="3DGS 训练监控看板")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--root", default="output", help="扫描哪个目录下的训练结果")
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    OUTPUT_ROOT = (ROOT / args.root).resolve() if not Path(args.root).is_absolute() else Path(args.root)
    print(f"扫描目录：{OUTPUT_ROOT}")
    runs = scan_runs()
    print(f"发现 {len(runs)} 个训练运行：" + ", ".join(r["name"] for r in runs[:8]))
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"看板地址：http://{args.host}:{args.port}/")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
