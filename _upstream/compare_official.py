"""同一个模型、同一个相机，官方 CUDA 光栅化器 vs 本实现：
   1) 图像是否一致（最强形式的正确性验证）
   2) 速度差多少
"""
import sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "_upstream" / "diff-gaussian-rasterization-main"))

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from src.colmap_dataset import ColmapDataset
from src.gaussian_model import GaussianModel
from src.rasterizer import build_cov3d, compute_colors, project_gaussians, rasterize

dev = torch.device("cuda")
K = 64
ds = ColmapDataset(str(ROOT / "data" / "real"), device=dev, downscale=2)
model = GaussianModel.load_ply(ROOT / "output" / "real3" / "ckpt" / "iter_01500.ply",
                               sh_degree=3, device=dev)
cam = ds.camera(ds.test_indices[0])
W, H = ds.width, ds.height
N = model.num_gaussians
print(f"模型 {N} 高斯，相机 {W}x{H}")

# ---- 本实现 ----
with torch.no_grad():
    t0 = time.time()
    for _ in range(10):
        colors = compute_colors(model.get_xyz, model.get_features, cam, 3)
        cov3d = build_cov3d(model.get_rotation, model.get_scaling)
        proj = project_gaussians(model.get_xyz, cov3d, cam)
        mine = rasterize(proj, colors, model._opacity, cam, ds.background, K=K).image
    torch.cuda.synchronize()
    t_mine = (time.time() - t0) / 10 * 1000

# ---- 官方 CUDA ----
# 完全按官方 graphics_utils 的约定构造，不自己推
import math
sys.path.insert(0, str(ROOT / "_upstream"))
from graphics_utils import getProjectionMatrix, getWorld2View2

R_cw = cam.R.cpu().numpy()                       # 本实现的 R 是 world->cam
T_cw = cam.T.cpu().numpy()
wvt = torch.from_numpy(getWorld2View2(R_cw.T, T_cw)).to(dev).transpose(0, 1).contiguous()
fovX = 2 * math.atan(W / (2 * cam.fx))
fovY = 2 * math.atan(H / (2 * cam.fy))
pm = getProjectionMatrix(0.01, 100.0, fovX, fovY).to(dev).transpose(0, 1).contiguous()
viewmat, full_proj = wvt, (wvt @ pm).contiguous()
fx, fy = cam.fx, cam.fy
znear, zfar = 0.01, 100.0

settings = GaussianRasterizationSettings(
    image_height=H, image_width=W,
    tanfovx=W / (2 * fx), tanfovy=H / (2 * fy),
    bg=ds.background.float(), scale_modifier=1.0,
    viewmatrix=viewmat, projmatrix=full_proj,
    sh_degree=3, campos=cam.camera_center.float(),
    prefiltered=False, debug=False,
)

def run_official():
    rast = GaussianRasterizer(raster_settings=settings)
    return rast(means3D=model.get_xyz, means2D=torch.zeros_like(model.get_xyz),
                shs=model.get_features, colors_precomp=None,
                opacities=model._opacity, scales=model.get_scaling,
                rotations=model.get_rotation, cov3D_precomp=None)

with torch.no_grad():
    img_o, radii = run_official()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        run_official()
    torch.cuda.synchronize()
    t_off = (time.time() - t0) / 10 * 1000

diff = (mine - img_o).abs()
print(f"\n速度：本实现 {t_mine:.1f} ms   官方 CUDA {t_off:.1f} ms   倍率 {t_mine/max(t_off,1e-9):.1f}x")

# K 越大差异应当越小——若如此，差异就来自 K 截断（官方无上限）
print("\nK 对差异的影响：")
for k in (16, 64, 128, 256):
    try:
        with torch.no_grad():
            torch.cuda.empty_cache()
            c2 = compute_colors(model.get_xyz, model.get_features, cam, 3)
            v2 = build_cov3d(model.get_rotation, model.get_scaling)
            p2 = project_gaussians(model.get_xyz, v2, cam)
            m2 = rasterize(p2, c2, model._opacity, cam, ds.background, K=k).image
        d2 = (m2 - img_o).abs()
        print(f"  K={k:<5} mean diff {float(d2.mean()):.5f}  max {float(d2.max()):.4f}  "
              f"显存 {torch.cuda.max_memory_allocated()/1e6:.0f} MB")
        del m2, d2
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"  K={k:<5} 失败：{str(e)[:60]}")
        torch.cuda.empty_cache()
print(f"图像差异：max {float(diff.max()):.4f}  mean {float(diff.mean()):.5f}")
print(f"  本实现 均值 {float(mine.mean()):.4f}   官方 均值 {float(img_o.mean()):.4f}")

from PIL import Image
out = ROOT / "output" / "_show" / "official_vs_mine.png"
out.parent.mkdir(parents=True, exist_ok=True)
d3 = (diff / max(float(diff.max()), 1e-6)).mean(dim=0, keepdim=True).repeat(3, 1, 1)
row = torch.cat([ds.image(ds.test_indices[0]), mine.clamp(0, 1), img_o.clamp(0, 1), d3], dim=2)
Image.fromarray((row.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(out)
print(f"对比图（GT | 本实现 | 官方 | 差异）-> {out}")
