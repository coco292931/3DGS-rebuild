"""干净的分段计时：forward / +backward / +optimizer / 完整训练步。

之前的版本用 losses.pop() 复用了同一张图，测的是「重复 backward」，
结论不可信，全部推倒重测。
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import SyntheticDataset
from src.gaussian_model import GaussianModel
from src.train import render_view

dev = torch.device("cuda")
K = int(sys.argv[1]) if len(sys.argv) > 1 else 20
ds = SyntheticDataset("data/synthetic", device=dev, downscale=1)
model = GaussianModel(sh_degree=3, device=dev)
model.init_from_points(ds.points, ds.point_colors, scale_factor=0.5)
model.active_sh_degree = 3
model.training_setup()
idx = ds.train_indices[0]
cam, gt, bg = ds.camera(idx), ds.image(idx), ds.background


def bench(fn, n=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000.0


def fwd():
    out, proj = render_view(model, cam, bg, K)
    return out


def fb():
    out, proj = render_view(model, cam, bg, K)
    (out.image - gt).abs().mean().backward()
    model.optimizer.zero_grad(set_to_none=True)


def fbs():
    out, proj = render_view(model, cam, bg, K)
    (out.image - gt).abs().mean().backward()
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)


with torch.no_grad():
    t_fwd = bench(fwd)
t_fb = bench(fb)
t_fbs = bench(fbs)

# 训练循环里额外的纯 CPU 动作（densification 统计等）
with torch.no_grad():
    out, proj = render_view(model, cam, bg, K)
    t_stats = bench(lambda: model.add_densification_stats(
        torch.zeros(model.num_gaussians, 2, device=dev), proj.valid))

print(f"K={K}  高斯 {model.num_gaussians}  片元 {out.num_fragments}  ({(out.num_fragments/(ds.width*ds.height)):.1f}/像素)")
print(f"  forward only          {t_fwd:8.2f} ms")
print(f"  forward + backward    {t_fb:8.2f} ms   (backward 净增 {t_fb - t_fwd:.2f} ms)")
print(f"  + optimizer.step      {t_fbs:8.2f} ms   (step 净增 {t_fbs - t_fb:.2f} ms)")
print(f"  完整步合计            {t_fbs:8.2f} ms  ->  {1000/t_fbs:.2f} iter/s")
print(f"  densify stats(参考)   {t_stats:8.2f} ms")
print(f"  实测训练速度约 1.1~1.4 iter/s -> 每步约 700~900 ms，与上面合计对比即可看出差异来源")
