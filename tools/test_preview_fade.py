"""预览遮罩效果对比：相机贴近物体时，有/无球形淡出的差别。"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.camera import Camera
from src.colmap_dataset import ColmapDataset
from src.gaussian_model import GaussianModel
from src.rasterizer import build_cov3d, compute_colors, near_fade_scale, project_gaussians, rasterize
from src.rasterizer_cuda import render_official

dev = torch.device("cuda")
ds = ColmapDataset(str(ROOT / "data" / "real"), device=dev, downscale=2)
m = GaussianModel.load_ply(ROOT / "output" / "cuda_long" / "ckpt" / "iter_30000.ply",
                           sh_degree=3, device=dev)

pc = ds.points.detach().float().cpu().numpy().astype(np.float64)
med = np.median(pc, axis=0)
p85 = float(np.percentile(np.linalg.norm(pc - med, axis=1), 85))
print(f"点云 p85 半径 = {p85:.3f}  （查看器与本脚本用同一基准）")
print(f"淡出半径 = p85 * 0.22 = {p85 * 0.22:.3f}")

idx = ds.test_indices[len(ds.test_indices) // 2]
base = ds.camera(idx)
gt = ds.image(idx)

# 造几个「相机推到很靠近物体」的位置：沿视线把相机朝目标挪
rows = []
for frac, label in ((0.0, "原视角"), (0.55, "推近 55%"), (0.78, "推近 78%"), (0.90, "推近 90%")):
    c = base.camera_center
    t = torch.tensor(med, device=dev, dtype=c.dtype)
    new_c = c + (t - c) * frac
    cam = Camera(R=base.R, T=base.T, width=base.width, height=base.height,
                 fx=base.fx, fy=base.fy, cx=base.cx, cy=base.cy)
    # 用 look-at 重建：把相机放到 new_c，仍看向场景中心
    up = torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=c.dtype)
    f = (t - new_c); f = f / f.norm()
    r = torch.cross(f, up); r = r / (r.norm() + 1e-9)
    u = torch.cross(r, f)
    R_wc = torch.stack([r, u, f], dim=0)          # cam->world
    cam = Camera(R=R_wc.T.contiguous(), T=(-R_wc @ new_c).contiguous(),
                 width=base.width, height=base.height,
                 fx=base.fx, fy=base.fy, cx=base.cx, cy=base.cy)

    outs = {}
    for tag, fs in (("off", 0.0), ("on", 0.22)):
        with torch.no_grad():
            o, _, _ = render_official(m.get_xyz, m.get_rotation, m.get_scaling,
                                      m._opacity, m.get_features, cam, ds.background,
                                      sh_degree=3, scene_radius=p85, fade_scale=fs)
            outs[tag] = o.image.clamp(0, 1)
    d = float((outs["on"] - outs["off"]).abs().max())
    print(f"{label:>10}: 有/无遮罩最大差异 {d:.4f}")
    rows.append((label, outs["off"], outs["on"], cam))

out = ROOT / "output" / "_show" / "fade_compare.png"
out.parent.mkdir(parents=True, exist_ok=True)
tiles = []
for label, off, on, _ in rows:
    tiles.append(torch.cat([gt, off, on], dim=2))
img = torch.cat(tiles, dim=1).permute(1, 2, 0).cpu().numpy()
Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(out)
print("\n每行：GT | 无遮罩 | 有遮罩（从左到右距离递增）")
print("->", out)
