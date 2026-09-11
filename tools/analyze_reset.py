"""opacity reset 前后的恢复过程 + 各 checkpoint 实测 PSNR（验证非线性）。"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

run = sys.argv[1] if len(sys.argv) > 1 else "output/cuda_long"
data = sys.argv[2] if len(sys.argv) > 2 else "data/real"

# ---- 1) loss 曲线看 reset 恢复 ----
log = ROOT / run / "train_log.jsonl"
rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
it = np.array([r["iter"] for r in rows], dtype=float)
loss = np.array([r["loss"] for r in rows], dtype=float)
print(f"=== {run} 的 loss 曲线（{len(rows)} 点，间隔 {int(it[1]-it[0])} 步）===")
for i in range(len(it)):
    bar = "#" * int(loss[i] / loss.max() * 50)
    print(f"  {int(it[i]):6d}  {loss[i]:.5f}  {bar}")

# ---- 2) 各 checkpoint 实测 PSNR ----
from src.colmap_dataset import ColmapDataset
from src.gaussian_model import GaussianModel
from src.metrics import psnr, ssim
from src.rasterizer_cuda import render_official

dev = torch.device("cuda")
ds = ColmapDataset(str(ROOT / data), device=dev, downscale=2)
cks = sorted((ROOT / run / "ckpt").glob("*.ply"),
             key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))
print(f"\n=== 各 checkpoint 实测（{len(ds.test_indices)} 个测试视角）===")
print(f"{'checkpoint':>14} {'PSNR':>8} {'SSIM':>8}   相对上一档")
prev = None
for c in cks:
    m = GaussianModel.load_ply(c, sh_degree=3, device=dev)
    ps, ss = [], []
    for idx in ds.test_indices[:12]:
        cam = ds.camera(idx)
        with torch.no_grad():
            o, _, _ = render_official(m.get_xyz, m.get_rotation, m.get_scaling,
                                      m._opacity, m.get_features, cam, ds.background, sh_degree=3)
            img = o.image.clamp(0, 1)
        gt = ds.image(idx)
        ps.append(psnr(img, gt)); ss.append(ssim(img, gt))
    np_, ns = float(np.mean(ps)), float(np.mean(ss))
    delta = "" if prev is None else f"{np_-prev:+.3f} dB"
    print(f"{c.stem:>14} {np_:8.3f} {ns:8.4f}   {delta}")
    prev = np_
