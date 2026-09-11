"""光栅化器正确性验证。

分四层：
    1. 几何：单个各向同性高斯的 2D 半径/位置是否符合解析预期
    2. 一致性：向量化实现 vs 朴素参考实现
    3. 语义：深度排序方向、遮挡关系、背景透射
    4. 梯度：有限差分 vs 反向传播

跑法： python -m pytest tests/test_rasterizer.py -v
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.camera import Camera, make_camera  # noqa: E402
from src.rasterizer import (  # noqa: E402
    build_cov3d,
    project_gaussians,
    render_gaussians,
    render_naive,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


def _camera(W=64, H=48, z=4.0, fov=60.0):
    cam = make_camera(W, H, fov, device=DEVICE)
    cam = cam.look_at(eye=(0.0, 0.0, z), target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0))
    return cam.to(DEVICE)


def _rand_gaussians(n, seed=0, spread=1.0, scale_range=(0.05, 0.25)):
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = (torch.rand(n, 3, generator=g) - 0.5) * 2 * spread
    # 深度必须打散：若所有高斯深度精确相等，深度序会退化成「扰动哪个哪个跳队首」，
    # 有限差分必然发散，那测的是用例的退化而不是实现。
    means[:, 2] = (torch.rand(n, generator=g) - 0.5) * 0.4
    quats = torch.randn(n, 4, generator=g)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    lo, hi = scale_range
    scales = torch.rand(n, 3, generator=g) * (hi - lo) + lo
    opacities = torch.rand(n, 1, generator=g) * 0.7 + 0.3
    colors = torch.rand(n, 3, generator=g)
    return [t.to(DEVICE, DTYPE) for t in (means, quats, scales, opacities, colors)]


def test_single_gaussian_geometry():
    """单个各向同性高斯，正对相机，2D 半径应约等于 3 * fx * s / z。"""
    cam = _camera(z=4.0)
    means = torch.zeros(1, 3, device=DEVICE, dtype=DTYPE)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEVICE, dtype=DTYPE)
    s = 0.1
    scales = torch.full((1, 3), s, device=DEVICE, dtype=DTYPE)

    cov3d = build_cov3d(quats, scales)
    proj = project_gaussians(means, cov3d, cam)

    # 解析半径 = 3 * sqrt(sigma_2d + COV2D_EPS)，低通项不能漏
    expected_radius = 3.0 * math.sqrt((cam.fx * s / 4.0) ** 2 + 0.3)
    assert torch.allclose(proj.radius, torch.tensor(expected_radius, device=DEVICE), rtol=0.05), (
        proj.radius.item(),
        expected_radius,
    )
    assert torch.allclose(proj.px, torch.tensor(cam.cx, device=DEVICE), atol=1e-4)
    assert torch.allclose(proj.py, torch.tensor(cam.cy, device=DEVICE), atol=1e-4)
    assert torch.allclose(proj.depth, torch.tensor(4.0, device=DEVICE), atol=1e-5)


def test_fast_matches_naive():
    """向量化分桶实现必须复现朴素实现的图像。"""
    cam = _camera(W=64, H=48)
    n = 40
    means, quats, scales, opacities, colors = _rand_gaussians(n, seed=1, spread=1.2)
    bg = torch.tensor([0.1, 0.2, 0.3], device=DEVICE, dtype=DTYPE)

    naive = render_naive(means, quats, scales, opacities, colors, cam, bg)

    cov3d = build_cov3d(quats, scales)
    proj = project_gaussians(means, cov3d, cam)
    from src.rasterizer import rasterize

    fast = rasterize(proj, colors, opacities, cam, bg, K=64).image

    diff = (fast - naive).abs().max().item()
    assert diff < 1e-5, f"向量化实现与朴素实现不一致，最大偏差 {diff:.2e}"


def test_sigma_clip_error_is_quantified():
    """3-sigma 包围盒裁剪是主动引入的近似，这里把它的大小钉死。

    向量化实现只在高斯的 3-sigma 包围盒内累加，朴素实现在这种模式下也是如此；
    若关掉裁剪、保留高斯尾部，图像会变多少——这个数字就是该近似的代价。
    将来谁改了半径系数（比如 3.0 -> 2.0），这个测试会立刻报警。
    """
    cam = _camera(W=64, H=48)
    means, quats, scales, opacities, colors = _rand_gaussians(40, seed=5, spread=1.2)
    bg = torch.tensor([0.1, 0.2, 0.3], device=DEVICE, dtype=DTYPE)

    full = render_naive(means, quats, scales, opacities, colors, cam, bg, clip_to_bbox=False)
    clipped = render_naive(means, quats, scales, opacities, colors, cam, bg, clip_to_bbox=True)

    max_err = (full - clipped).abs().max().item()
    mean_err = (full - clipped).abs().mean().item()
    print(f"\n[3σ 裁剪] 最大偏差 {max_err:.4f} / 平均偏差 {mean_err:.4f}")
    assert max_err < 0.05, f"3σ 裁剪误差过大（{max_err:.4f}），半径系数可能被改错了"


def test_k_truncation_keeps_nearest():
    """K=1 时每个像素只能保留最近的一个高斯。"""
    cam = _camera(z=4.0)
    means = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=DEVICE, dtype=DTYPE)
    means[1, 2] = -0.5
    quats = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]], device=DEVICE, dtype=DTYPE)
    scales = torch.full((2, 3), 0.15, device=DEVICE, dtype=DTYPE)
    opacities = torch.full((2, 1), 5.0, device=DEVICE, dtype=DTYPE)
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=DEVICE, dtype=DTYPE)
    bg = torch.zeros(3, device=DEVICE, dtype=DTYPE)

    from src.rasterizer import rasterize

    cov3d = build_cov3d(quats, scales)
    proj = project_gaussians(means, cov3d, cam)
    img = rasterize(proj, colors, opacities, cam, bg, K=1).image
    cy, cx = int(cam.cy), int(cam.cx)
    r, _, b = img[:, cy, cx].tolist()
    assert r > 0.9 and b < 0.1, f"K=1 未保留最近高斯 (r={r:.3f}, b={b:.3f})"


def test_depth_ordering():
    """两个完全重叠的高斯：近处红色应盖住远处蓝色。"""
    cam = _camera(z=4.0)
    means = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=DEVICE, dtype=DTYPE)
    # 让第二个高斯离相机更远（相机在 z=+4 看向原点，所以 z 更小 = 更远）
    means[1, 2] = -0.5
    quats = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]], device=DEVICE, dtype=DTYPE)
    scales = torch.full((2, 3), 0.15, device=DEVICE, dtype=DTYPE)
    opacities = torch.full((2, 1), 5.0, device=DEVICE, dtype=DTYPE)  # sigmoid(5)≈0.993
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=DEVICE, dtype=DTYPE)
    bg = torch.zeros(3, device=DEVICE, dtype=DTYPE)

    img = render_naive(means, quats, scales, opacities, colors, cam, bg)
    cy, cx = int(cam.cy), int(cam.cx)
    r, g, b = img[:, cy, cx].tolist()
    assert r > b, f"深度顺序错误：近处红色没盖住远处蓝色 (r={r:.3f}, b={b:.3f})"


def test_background_when_everything_culled():
    """全部高斯在相机后方时应输出纯背景。"""
    cam = _camera(z=4.0)
    means = torch.tensor([[0.0, 0.0, 100.0], [0.0, 0.0, 100.0]], device=DEVICE, dtype=DTYPE)
    quats = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]], device=DEVICE, dtype=DTYPE)
    scales = torch.full((2, 3), 0.15, device=DEVICE, dtype=DTYPE)
    opacities = torch.full((2, 1), 2.0, device=DEVICE, dtype=DTYPE)
    shs = torch.zeros(2, 16, 3, device=DEVICE, dtype=DTYPE)
    bg = torch.tensor([0.25, 0.5, 0.75], device=DEVICE, dtype=DTYPE)

    out = render_gaussians(means, quats, scales, opacities, shs, cam, bg, sh_degree=3, K=8)
    expected = bg.reshape(3, 1, 1).expand(3, cam.height, cam.width)
    assert torch.allclose(out.image, expected, atol=1e-6)


def _numeric_grad(loss_of, params, name, eps):
    """对某个参数做中心差分。loss_of 接收参数字典并返回标量 loss。"""
    base = params[name].detach()
    grad = torch.zeros(base.numel(), device=base.device, dtype=base.dtype)
    for i in range(base.numel()):
        orig = base.reshape(-1)[i].item()
        for sign, slot in ((+eps, 0), (-eps, 1)):
            perturbed = dict(params)
            p = base.clone().reshape(-1)
            p[i] = orig + sign
            perturbed[name] = p.reshape(base.shape)
            with torch.no_grad():
                val = loss_of(perturbed).item()
            if slot == 0:
                fp = val
            else:
                fm = val
        grad[i] = (fp - fm) / (2 * eps)
    return grad.reshape(base.shape)


def test_gradients_match_finite_difference():
    """反向传播必须与有限差分吻合（不靠 autograd 自证）。

    必须在 float64 下做。float32 时 loss 的变化量已淹没在浮点精度里，
    有限差分本身失效（实测：双精度 eps=1e-4 相对误差 6e-10；
    同样 eps 在 float32 下退化到 0.5 量级——那是差分方法的问题，不是实现的问题）。
    """
    cam = _camera(W=32, H=24)
    n = 6
    means, quats, scales, opacities, colors = [t.double() for t in _rand_gaussians(n, seed=7, spread=0.8)]
    bg = torch.zeros(3, device=DEVICE, dtype=torch.float64)
    target = torch.rand(3, cam.height, cam.width, device=DEVICE, dtype=torch.float64)

    def loss_of(tensors):
        cov3d = build_cov3d(tensors["quats"], tensors["scales"])
        proj = project_gaussians(tensors["means"], cov3d, cam)
        from src.rasterizer import rasterize

        img = rasterize(proj, tensors["colors"], tensors["opacities"], cam, bg, K=32).image
        return ((img - target) ** 2).mean()

    params = {"means": means, "quats": quats, "scales": scales, "opacities": opacities, "colors": colors}
    for k in params:
        params[k] = params[k].clone().detach().requires_grad_(True)

    loss_of(params).backward()

    # eps 取 1e-5：双精度下中心差分的最优尺度（~eps^(1/3)），
    # 再小会被舍入噪声吃掉，再大则会跨过 3-sigma 包围盒的取整台阶。
    checks = [
        ("opacities", 1e-5, 1e-6),
        ("colors", 1e-5, 1e-6),
        ("scales", 1e-5, 1e-4),
        ("means", 1e-5, 1e-4),
    ]
    for name, eps, tol in checks:
        num = _numeric_grad(loss_of, params, name, eps)
        ana = params[name].grad
        denom = torch.maximum(num.abs(), ana.abs()).clamp(min=1e-6)
        rel = ((num - ana).abs() / denom).max().item()
        assert rel < tol, f"{name} 梯度不匹配，最大相对误差 {rel:.3e}"


def test_training_step_reduces_loss():
    """端到端冒烟测试：对一张目标图做几十步优化，loss 必须显著下降。"""
    cam = _camera(W=48, H=36)
    n = 60
    means, quats, scales, opacities, colors = _rand_gaussians(n, seed=3, spread=1.0)
    bg = torch.zeros(3, device=DEVICE, dtype=DTYPE)
    target = render_naive(means, quats, scales, opacities, colors, cam, bg).detach()

    params = {
        "means": means.clone().detach().requires_grad_(True),
        "quats": quats.clone().detach().requires_grad_(True),
        "scales": scales.clone().detach().requires_grad_(True),
        "opacities": opacities.clone().detach().requires_grad_(True),
        "colors": torch.zeros(n, 3, device=DEVICE, dtype=DTYPE).requires_grad_(True),
    }
    opt = torch.optim.Adam([{"params": [params["means"]], "lr": 1e-3},
                            {"params": [params["quats"], params["scales"], params["opacities"], params["colors"]], "lr": 1e-2}])

    from src.rasterizer import rasterize

    def step():
        cov3d = build_cov3d(params["quats"], params["scales"])
        proj = project_gaussians(params["means"], cov3d, cam)
        return rasterize(proj, params["colors"], params["opacities"], cam, bg, K=32).image

    with torch.no_grad():
        first = ((step() - target) ** 2).mean().item()
    for _ in range(60):
        opt.zero_grad()
        loss = ((step() - target) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        last = ((step() - target) ** 2).mean().item()

    assert last < first * 0.5, f"优化没有收敛：{first:.5f} -> {last:.5f}"
