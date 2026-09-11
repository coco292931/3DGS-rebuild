"""GaussianModel 冒烟测试：初始化、渲染、PLY 往返、密度控制。"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.camera import make_camera  # noqa: E402
from src.gaussian_model import GaussianModel  # noqa: E402
from src.rasterizer import render_gaussians  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _sphere_points(n=400, radius=0.5, seed=0):
    g = np.random.default_rng(seed)
    v = g.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pts = torch.tensor(v * radius, dtype=torch.float32)
    colors = torch.tensor(v * 0.5 + 0.5, dtype=torch.float32)
    return pts, colors


def _camera(W=64, H=48):
    cam = make_camera(W, H, 60.0, device=DEVICE)
    return cam.look_at(eye=(0.0, 0.0, 2.5), target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)).to(DEVICE)


def _render(model, cam):
    bg = torch.zeros(3, device=DEVICE)
    return render_gaussians(
        model.get_xyz, model.get_rotation, model.get_scaling,
        model._opacity, model.get_features, cam, bg,
        sh_degree=model.active_sh_degree, K=16,
    ).image


def test_init_and_render():
    pts, colors = _sphere_points()
    model = GaussianModel(sh_degree=3, device=DEVICE)
    model.init_from_points(pts, colors)
    assert model.num_gaussians == 400

    img = _render(model, _camera())
    assert img.shape == (3, 48, 64)
    assert img.max().item() > 0.05, "球面点云渲染出来是纯黑，初始化或光栅化有问题"


def test_ply_roundtrip(tmp_path=None):
    pts, colors = _sphere_points()
    model = GaussianModel(sh_degree=3, device=DEVICE)
    model.init_from_points(pts, colors)
    # 给参数一点扰动，避免「全零也一致」的假通过
    with torch.no_grad():
        model._features_rest.normal_(0, 0.1)
        model._rotation.normal_(0, 0.2)
        model._scaling.add_(torch.randn_like(model._scaling) * 0.05)

    path = Path(__file__).resolve().parent / "_roundtrip.ply"
    model.save_ply(path)
    try:
        loaded = GaussianModel.load_ply(path, sh_degree=3, device=DEVICE)
        assert loaded.num_gaussians == model.num_gaussians
        for name in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"):
            a = getattr(model, name).detach()
            b = getattr(loaded, name).detach()
            assert a.shape == b.shape, f"{name} 形状不一致 {tuple(a.shape)} vs {tuple(b.shape)}"
            assert torch.allclose(a, b, atol=1e-6), f"{name} PLY 往返后不一致"

        cam = _camera()
        assert torch.allclose(_render(model, cam), _render(loaded, cam), atol=1e-6), "往返后渲染结果不一致"
    finally:
        path.unlink(missing_ok=True)


def test_densify_clone_and_split_counts():
    pts, colors = _sphere_points(n=200)
    model = GaussianModel(sh_degree=3, device=DEVICE)
    model.init_from_points(pts, colors)
    model.scene_extent = 1.0

    n0 = model.num_gaussians
    # 人为制造梯度：一半高斯给大梯度，且缩放一小一大，分别触发 clone 和 split
    grads = torch.zeros(n0, 1, device=DEVICE)
    grads[:100] = 1.0
    with torch.no_grad():
        model._scaling[:50] = torch.log(torch.full((50, 3), 0.001, device=DEVICE))   # 小 -> clone
        model._scaling[50:100] = torch.log(torch.full((50, 3), 0.5, device=DEVICE))  # 大 -> split

    model.densify_and_clone(grads, grad_threshold=2e-4)
    assert model.num_gaussians == n0 + 50, f"clone 数量不对：{model.num_gaussians}"

    # clone 之后高斯数变了，梯度向量的长度必须重建（clone 把新点追加在末尾，前 100 个索引不变）
    grads2 = torch.zeros(model.num_gaussians, 1, device=DEVICE)
    grads2[:100] = 1.0
    model.densify_and_split(grads2, grad_threshold=2e-4, n_split=2, seed=0)
    # 被 split 的是那 50 个「大尺度且梯度大」的高斯：-50 +50*2 = +50
    assert model.num_gaussians == n0 + 50 + 50, f"split 数量不对：{model.num_gaussians}"


def test_opacity_reset_and_prune():
    pts, colors = _sphere_points(n=100)
    model = GaussianModel(sh_degree=3, device=DEVICE)
    model.init_from_points(pts, colors)

    with torch.no_grad():
        model._opacity.fill_(5.0)  # sigmoid(5)≈0.993
    model.reset_opacity(max_opacity=0.01)
    assert model.get_opacity.max().item() <= 0.0101, "opacity reset 没把不透明度压下去"

    before = model.num_gaussians
    model.prune(min_opacity=0.5)
    assert model.num_gaussians < before, "prune 没有剪掉任何高斯"
