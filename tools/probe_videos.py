"""侦察候选视频：分辨率、时长、帧率，并抽帧拼图供人眼判断是否适合 SfM。"""
import json
import subprocess
import sys
from pathlib import Path

FF = r"C:\Users\tonyp\Downloads\FFmpeg\bin\ffmpeg.exe"
FFPROBE = r"C:\Users\tonyp\Downloads\FFmpeg\bin\ffprobe.exe"

cands = json.loads(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
for i, v in enumerate(cands):
    p = Path(v)
    if not p.exists():
        rows.append({"path": v, "error": "missing"})
        continue
    try:
        r = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
             "-of", "json", str(p)],
            capture_output=True, text=True, timeout=30,
        )
        st = json.loads(r.stdout)["streams"][0]
        dur = float(st.get("duration") or 0)
        info = {
            "path": str(p),
            "name": p.name,
            "size_MB": round(p.stat().st_size / 1e6, 1),
            "res": f"{st['width']}x{st['height']}",
            "fps": st.get("r_frame_rate"),
            "dur_s": round(dur, 1),
        }
    except Exception as e:
        info = {"path": str(p), "name": p.name, "error": repr(e)[:80]}
    # 抽 3 帧
    try:
        ts = [max(0.1, (info.get("dur_s", 10) or 10) * f) for f in (0.15, 0.5, 0.85)]
        files = []
        for j, t in enumerate(ts):
            outp = out_dir / f"v{i}_{j}.png"
            subprocess.run([FF, "-y", "-v", "error", "-ss", str(t), "-i", str(p),
                            "-frames:v", "1", "-vf", "scale=320:-1", str(outp)],
                           capture_output=True, timeout=60)
            if outp.exists():
                files.append(str(outp))
        info["frames"] = files
    except Exception as e:
        info["frames"] = []
    rows.append(info)
    print(json.dumps(info, ensure_ascii=False), flush=True)

# 拼图
try:
    from PIL import Image, ImageDraw
    tiles = []
    for info in rows:
        fs = info.get("frames", [])
        if not fs:
            continue
        ims = [Image.open(f).convert("RGB") for f in fs]
        h = max(im.height for im in ims)
        w = sum(im.width for im in ims)
        strip = Image.new("RGB", (w, h + 22), (20, 20, 24))
        x = 0
        for im in ims:
            strip.paste(im, (x, 22)); x += im.width
        ImageDraw.Draw(strip).text((6, 5), info.get("name", "?")[:60] + "  " + info.get("res", ""), fill=(255, 230, 120))
        tiles.append(strip)
    if tiles:
        W = max(t.width for t in tiles)
        H = sum(t.height for t in tiles)
        grid = Image.new("RGB", (W, H), (12, 12, 16))
        y = 0
        for t in tiles:
            grid.paste(t, (0, y)); y += t.height
        grid.save(out_dir / "video_grid.png")
        print("grid ->", out_dir / "video_grid.png")
except Exception as e:
    print("grid failed:", e)
