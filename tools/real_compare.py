"""真实场景：GT 与渲染并排对比。"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.colmap_dataset import ColmapDataset
from src.gaussian_model import GaussianModel
from src.metrics import psnr
from src.rasterizer import render_gaussians

dev = torch.device("cuda")
K = 64
ds = ColmapDataset("data/real", device=dev, downscale=2)
model = GaussianModel.load_ply(sys.argv[1], sh_degree=3, device=dev)

picks = [ds.test_indices[0], ds.test_indices[len(ds.test_indices) // 2], ds.test_indices[-1]]
rows = []
for i in picks:
    cam = ds.camera(i)
    with torch.no_grad():
        out = render_gaussians(model.get_xyz, model.get_rotation, model.get_scaling,
                               model._opacity, model.get_features, cam, ds.background,
                               sh_degree=3, K=K)
        img = out.image.clamp(0, 1)
    gt = ds.image(i)
    p = psnr(img, gt)
    print(f"frame {i}: PSNR {p:.2f} dB")
    row = torch.cat([gt, img], dim=2).permute(1, 2, 0).cpu().numpy()
    rows.append((row, p))

H = sum(r.shape[0] for r, _ in rows)
W = max(r.shape[1] for r, _ in rows)
canvas = np.ones((H, W, 3), dtype=np.float32) * 0.1
y = 0
for row, _ in rows:
    canvas[y:y + row.shape[0], :row.shape[1]] = row
    y += row.shape[0]
Image.fromarray((canvas * 255 + 0.5).astype(np.uint8)).save(sys.argv[2])
print("saved", sys.argv[2])
