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
        # 导入还在跑的时候，sparse/ 可能只写了一半，这时报出的帧数是误导性的。
        # 用 worker 留下的标记文件把「处理中」和「已就绪」区分开。
        importing = (d / ".importing").exists()
        has_sfm = (d / "sparse").exists()
        n_view = 0
        if has_sfm and not importing:
            try:
                import pycolmap
                best = ColmapDataset._find_model(d)
                n_view = pycolmap.Reconstruction(str(best)).num_reg_images()
            except Exception:
                n_view = 0        # 读不出来就报 0，不要报 -1 让人以为是失败
        out.append({
            "name": d.name,
            "images": len(imgs),
            "has_sfm": has_sfm and not importing,
            "importing": importing,
            "registered": n_view,
            "kind": "importing" if importing else ("colmap" if has_sfm else ("images" if imgs else "empty")),
        })
    return out


def list_jobs() -> list[dict]:
    jobs = []
    for name, j in JOBS.items():
        if j.get("thread") is not None:          # 导入这类线程任务
            th = j["thread"]
            alive = th.is_alive()
            jobs.append({"id": name, "kind": j["kind"], "cmd": j["cmd_short"],
                         "alive": alive, "returncode": None, "started": j["started"]})
            continue
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
        iters = int(args.get("iters", 30000))
        ckpt = int(args.get("ckpt_interval", 5000))
        preview = int(args.get("preview_interval", 500))
        cmd = [
            sys.executable, "-u", "-m", "src.train",
            "--data", f"data/{ds}",
            "--backend", backend,
            "--iters", str(iters),
            "--downscale", str(args.get("downscale", 2)),
            "--out", f"output/{out_name}",
            "--init-scale-factor", str(args.get("init_scale", 1.0)),
            "--max-gaussians", str(args.get("max_gaussians", 60000)),
            # 官方是「前一半步数增密」。但步数填得很小时（比如 200 步试跑），
            # 一半只有 100 步，高斯还没长开就停了增密。给个下限，
            # 同时不超过总步数的 80%，免得最后完全没有收敛期。
            "--densify-until", str(max(500, min(int(iters * 0.5), int(iters * 0.8)))),
            "--opacity-reset-interval", str(args.get("opacity_reset", 3000)),
            "--ckpt-interval", str(ckpt),
            "--preview-interval", str(preview),
            "--log-interval", str(max(100, iters // 60)),
            "--eval-interval", str(max(500, iters // 6)),
        ]
        if backend == "torch":
            cmd += ["--K", str(args.get("K", 64)), "--max-fragments", "16000000"]
        parts = [f"训练 {ds}", backend, f"{iters} 步"]
        if ckpt: parts.append(f"ckpt/{ckpt}")
        if preview: parts.append(f"预览/{preview}")
        cmd_short = " · ".join(parts)
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



# ---------------------------------------------------------------------------
# 视频导入：扫描候选视频 -> 抽帧 -> SfM
# ---------------------------------------------------------------------------
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".MP4", ".MOV"}
SCAN_DIRS = [
    ROOT.parent,                       # Downloads
    ROOT.parent / "3d-rebuild",
    ROOT.parent / "Videos",
    ROOT / "videos",
]
FFMPEG = None
for _c in [Path(r"C:\Users\tonyp\Downloads\FFmpeg\bin\ffmpeg.exe"),
           Path("ffmpeg"), Path("ffmpeg.exe")]:
    if isinstance(_c, Path) and _c.exists():
        FFMPEG = str(_c)
        break
    if not isinstance(_c, Path):
        FFMPEG = str(_c)
        break


def scan_videos() -> list[dict]:
    """扫描常见目录下的视频，标注是否已经导入过。"""
    seen, out = set(), []
    imported = {d.name for d in (ROOT / "data").iterdir()} if (ROOT / "data").exists() else set()
    for base in SCAN_DIRS:
        if not base.exists():
            continue
        try:
            files = [p for p in base.iterdir() if p.is_file() and p.suffix in VIDEO_EXTS]
        except Exception:
            continue
        for p in files[:40]:
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            st = p.stat()
            # 猜一个数据集名：文件名去掉扩展、非字母数字压成下划线
            guess = re.sub(r"[^0-9A-Za-z_]+", "_", p.stem)[:40] or "video"
            out.append({
                "name": p.name,
                "path": str(p),
                "dir": base.name,
                "size_mb": round(st.st_size / 1e6, 1),
                "mtime": time.strftime("%m-%d %H:%M", time.localtime(st.st_mtime)),
                "guess_dataset": guess,
                "imported": guess in imported,
            })
    out.sort(key=lambda x: x["mtime"], reverse=True)
    return out[:40]


def probe_video(path: str) -> dict:
    """读分辨率/时长，用于导入前给用户确认。"""
    import subprocess

    ffprobe = str(Path(FFMPEG).with_name("ffprobe.exe")) if FFMPEG else "ffprobe"
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate",
             "-show_entries", "format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30)
        d = json.loads(r.stdout)
        st = d["streams"][0]
        return {"width": st["width"], "height": st["height"],
                "duration": round(float(d["format"]["duration"]), 1)}
    except Exception as e:
        return {"error": str(e)[:120]}


def import_video_worker(name: str, src: Path, fps: float, scale: int, log_path: Path):
    """在后台线程里跑完整导入。进度写进 log 文件，前端轮询显示。"""
    import subprocess

    ds = ROOT / "data" / name
    frames = ds / "frames"
    marker = ds / ".importing"
    ds.mkdir(parents=True, exist_ok=True)
    marker.write_text("1", encoding="utf-8")
    frames.mkdir(parents=True, exist_ok=True)
    for old in frames.glob("*.png"):
        old.unlink()

    fh = open(log_path, "w", encoding="utf-8", buffering=1)

    def say(msg):
        fh.write(msg + "\n")
        fh.flush()

    def run(cmd, label):
        say(f"[{time.strftime('%H:%M:%S')}] {label} …")
        p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        tail = ((p.stdout or "") + (p.stderr or "")).strip()[-400:]
        if tail:
            say("    " + tail.replace("\n", "\n    "))
        if p.returncode != 0:
            raise RuntimeError(f"{label} 失败（返回码 {p.returncode}）")

    try:
        say(f"开始导入 {src.name}")
        n = len(list(frames.glob("*.png")))
        # 抽帧。fps 必须够密——手持转半秒能转过二十几度，帧间视差过大会直接匹配失败。
        run([FFMPEG, "-y", "-v", "error", "-i", str(src),
             "-vf", f"fps={fps},scale={scale}:-1", "-q:v", "2",
             str(frames / "frame_%04d.png")], "抽帧")
        n = len(list(frames.glob("*.png")))
        say(f"    抽出 {n} 帧")
        if n < 12:
            raise RuntimeError(f"只抽出 {n} 帧，太少，无法重建")

        for stage, label in (("features", "特征提取"), ("match", "顺序匹配"), ("map", "增量式重建")):
            cmd = [sys.executable, "-u", "tools/run_sfm.py", f"data/{name}", "--stage", stage]
            if stage == "match":
                cmd.append("--sequential")
            run(cmd, label)

        n_reg = 0
        try:
            import pycolmap
            n_reg = pycolmap.Reconstruction(str(ColmapDataset._find_model(ds))).num_reg_images()
        except Exception as e:
            say(f"    读回模型失败：{str(e)[:120]}")
        say(f"[{time.strftime('%H:%M:%S')}] 完成：{n} 帧抽出，{n_reg} 帧注册")
    except Exception as e:
        say(f"[{time.strftime('%H:%M:%S')}] 失败：{e}")
    finally:
        marker.unlink(missing_ok=True)
        fh.close()


def start_import(args: dict) -> dict:
    """导入是长任务（几百帧要跑几分钟），必须放到后台，
    否则前端只能干等，中途刷新还会看到 sparse/ 写了一半的中间状态。"""
    import threading

    src = Path(args["path"])
    if not src.exists():
        raise FileNotFoundError(f"视频不存在：{src}")
    name = re.sub(r"[^0-9A-Za-z_]+", "_", args.get("name") or src.stem)[:40] or "video"
    fps = float(args.get("fps", 6))
    scale = int(args.get("scale", 1280))

    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = f"import_{time.strftime('%m%d_%H%M%S')}"
    log_path = JOBS_DIR / f"{job_id}.log"
    log_path.write_text("", encoding="utf-8")

    t = threading.Thread(target=import_video_worker, args=(name, src, fps, scale, log_path),
                         daemon=True)
    t.start()
    JOBS[job_id] = {"proc": None, "thread": t, "kind": "import", "log": str(log_path),
                    "started": time.strftime("%H:%M:%S"),
                    "cmd_short": f"导入 {src.name} → {name}"}
    return {"id": job_id, "dataset": name, "cmd": f"导入 {src.name}"}


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

        if path == "/api/import":
            if not body.get("path"):
                return self._json({"error": "缺少 path"}, 400)
            try:
                return self._json(start_import(body))
            except Exception as e:
                return self._json({"error": str(e)[:300]}, 500)

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

        if path == "/api/videos":
            return self._json({"videos": scan_videos(), "ffmpeg": FFMPEG})

        if path == "/api/previews":
            q = self.path.split("?", 1)[1] if "?" in self.path else ""
            name = dict(x.split("=", 1) for x in q.split("&") if "=" in x).get("name", "")
            d = OUTPUT_ROOT / name / "preview"
            files = sorted(d.glob("*.png"), key=_ckpt_iter) if d.exists() else []
            return self._json({"files": [f.name for f in files],
                               "latest": files[-1].name if files else None})

        if path == "/api/probe":
            q = self.path.split("?", 1)[1] if "?" in self.path else ""
            p = dict(x.split("=", 1) for x in q.split("&") if "=" in x).get("path", "")
            return self._json(probe_video(unquote(p)))

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
