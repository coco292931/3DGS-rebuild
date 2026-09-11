"""纯 PyTorch 可微高斯光栅化器。

这是整个复刻的核心。与官方 CUDA 实现（diff-gaussian-rasterization）的区别：

    官方    : 16x16 tile 分块 + 块内深度排序 + 每块限量，手写前反向 CUDA 核
    本实现  : 按像素分桶 + 每像素保留最近 K 个 + 稠密张量 alpha 合成，反向交给 autograd

取舍已明确：分桶方案没有 Python 循环、完全向量化、显存可控，
代价是高斯极端交叠处会丢弃第 K 个之后的贡献。K 取 8~16 时在中小场景里与
官方结果几乎不可区分（见 tests/test_rasterizer.py 与朴素实现的对比）。

数学约定：
    Sigma_3D = R S S^T R^T
    Sigma_2D = J W Sigma_3D W^T J^T     (J 为投影雅可比，W 为相机旋转)
    conic    = Sigma_2D^-1 = [a, b, c]，代表 [[a, b], [b, c]]
    alpha    = opacity * exp(-0.5 * (a*dx^2 + c*dy^2) - b*dx*dy)
"""

from dataclasses import dataclass

import torch

from .camera import Camera
from .sh import eval_sh

# 官方实现的低通滤波项：给 2D 协方差对角线加一个常量，
# 避免高斯缩到亚像素时出现锯齿闪烁。
COV2D_EPS = 0.3

# 每个像素最多保留多少个高斯贡献
DEFAULT_K = 12


@dataclass
class ProjectedGaussians:
    px: torch.Tensor        # [N] 屏幕 x
    py: torch.Tensor        # [N] 屏幕 y
    depth: torch.Tensor     # [N] 相机空间 z
    radius: torch.Tensor    # [N] 3-sigma 像素半径
    conic: torch.Tensor     # [N, 3] (a, b, c)
    valid: torch.Tensor     # [N] bool
    cam_xy: torch.Tensor    # [N, 2] 相机空间 (x, y)，密度控制读它的梯度（与官方一致）


@dataclass
class RenderOutput:
    image: torch.Tensor     # [3, H, W]
    num_fragments: int      # 展开后的片元总数（未截断）
    num_kept: int           # 实际参与合成的片元数（截断后）


def quat_to_rotmat(quats: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """四元数 [N,4] (w, x, y, z) -> 旋转矩阵 [N,3,3]。"""
    q = quats
    if normalize:
        q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = q.unbind(-1)
    two = 2.0
    rows = torch.stack(
        [
            1 - two * (y * y + z * z), two * (x * y - w * z), two * (x * z + w * y),
            two * (x * y + w * z), 1 - two * (x * x + z * z), two * (y * z - w * x),
            two * (x * z - w * y), two * (y * z + w * x), 1 - two * (x * x + y * y),
        ],
        dim=-1,
    )
    return rows.reshape(-1, 3, 3)


def build_cov3d(quats: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """构造 3D 协方差矩阵 [N,3,3] = R diag(s^2) R^T。

    scales 传进来的应当是已经过 exp 的正尺度。
    """
    R = quat_to_rotmat(quats)
    M = R * scales.unsqueeze(1)          # R @ diag(s)
    return M @ M.transpose(1, 2)


def compute_colors(
    means: torch.Tensor,
    shs: torch.Tensor,
    cam: Camera,
    sh_degree: int,
) -> torch.Tensor:
    """按视角求球谐颜色 [N,3]。方向取「相机 -> 高斯」。"""
    center = cam.camera_center.to(device=means.device, dtype=means.dtype)
    dirs = means - center.reshape(1, 3)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return eval_sh(sh_degree, shs, dirs)


def project_gaussians(
    means: torch.Tensor,
    cov3d: torch.Tensor,
    cam: Camera,
    max_radius_frac: float = 0.3,
) -> ProjectedGaussians:
    """把 3D 高斯投影到屏幕，返回 2D 均值/半径/conic。"""
    device, dtype = means.device, means.dtype
    viewmat = cam.world_view_transform.to(device=device, dtype=dtype)

    ones = torch.ones(means.shape[0], 1, device=device, dtype=dtype)
    p_h = torch.cat([means, ones], dim=1)
    p_cam = p_h @ viewmat.transpose(0, 1)
    tx, ty, tz = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]

    valid = tz > cam.znear
    tz_safe = torch.where(valid, tz, torch.ones_like(tz))

    n = means.shape[0]
    J = torch.zeros(n, 3, 3, device=device, dtype=dtype)
    J[:, 0, 0] = cam.fx / tz_safe
    J[:, 1, 1] = cam.fy / tz_safe
    J[:, 0, 2] = -cam.fx * tx / (tz_safe * tz_safe)
    J[:, 1, 2] = -cam.fy * ty / (tz_safe * tz_safe)

    W = viewmat[:3, :3]
    T = J @ W
    cov2d = T @ cov3d @ T.transpose(1, 2)

    a = cov2d[:, 0, 0] + COV2D_EPS
    b = 0.5 * (cov2d[:, 0, 1] + cov2d[:, 1, 0])
    c = cov2d[:, 1, 1] + COV2D_EPS

    det = torch.clamp(a * c - b * b, min=1e-9)
    conic = torch.stack([c / det, -b / det, a / det], dim=-1)

    mid = 0.5 * (a + c)
    disc = torch.clamp(mid * mid - det, min=0.0)
    lambda_max = torch.clamp(mid + torch.sqrt(disc), min=1e-9)
    radius = 3.0 * torch.sqrt(lambda_max)

    # 相机空间坐标必须是屏幕坐标的上游：密度控制要对它调用 retain_grad()，
    # 若只作为兄弟节点（事后 stack 出来），梯度不流经它，.grad 恒为 None。
    cam_xy = torch.stack([tx, ty], dim=-1)
    px = cam.fx * cam_xy[:, 0] / tz_safe + cam.cx
    py = cam.fy * cam_xy[:, 1] / tz_safe + cam.cy

    valid = valid & torch.isfinite(radius) & (radius > 1e-3)
    valid = valid & (radius < max_radius_frac * max(cam.width, cam.height))

    # 相机空间坐标一并返回：官方用它的梯度做密度控制。若改用屏幕坐标梯度，
    # 量纲会差 fx/z 倍（本场景约 38 倍），官方阈值 2e-4 就永远够不到。
    return ProjectedGaussians(px=px, py=py, depth=tz, radius=radius, conic=conic, valid=valid,
                              cam_xy=cam_xy)


def compute_bboxes(proj: ProjectedGaussians, cam: Camera):
    """每个高斯的整数屏幕包围盒，已 clamp 到图像内且保证 x1>=x0, y1>=y0。

    这是 3-sigma 裁剪的落地点：包围盒之外的像素不参与合成。
    向量化实现与朴素参考实现共用它，保证两者比较的是合成逻辑而非裁剪策略。
    """
    W, H = cam.width, cam.height
    px, py, radius = proj.px, proj.py, proj.radius
    x0 = torch.clamp(torch.floor(px - radius).long(), 0, W - 1)
    x1 = torch.clamp(torch.ceil(px + radius).long(), 0, W - 1)
    y0 = torch.clamp(torch.floor(py - radius).long(), 0, H - 1)
    y1 = torch.clamp(torch.ceil(py + radius).long(), 0, H - 1)
    return x0, torch.maximum(x1, x0), y0, torch.maximum(y1, y0)


_DUMMY_CACHE: dict = {}


def _dummy_indices(shape, n_gaussians: int, device):
    """给空槽位用的、均匀分散的合法索引（只依赖形状，可缓存）。"""
    key = (shape, str(device))
    base = _DUMMY_CACHE.get(key)
    if base is None:
        base = torch.arange(shape[0] * shape[1], device=device, dtype=torch.long).reshape(shape)
        _DUMMY_CACHE[key] = base
    return base % n_gaussians


def _build_fragments(proj: ProjectedGaussians, cam: Camera, device):
    """把每个高斯的屏幕包围盒展开成片元列表（像素 id + 高斯 id + 深度）。"""
    W, H = cam.width, cam.height
    valid = proj.valid
    x0, x1, y0, y1 = compute_bboxes(proj, cam)

    counts = (x1 - x0 + 1) * (y1 - y0 + 1)
    counts = torch.where(valid, counts, torch.zeros_like(counts))
    total = int(counts.sum().item())
    if total == 0:
        return None

    gids = torch.repeat_interleave(
        torch.arange(valid.shape[0], device=device), counts, output_size=total
    )
    starts = torch.cumsum(counts, dim=0) - counts
    offsets = torch.arange(total, device=device) - starts[gids]

    widths = (x1 - x0 + 1)[gids]
    frag_y = y0[gids] + offsets // widths
    frag_x = x0[gids] + offsets % widths

    return gids, frag_y * W + frag_x


def near_fade_scale(means: torch.Tensor, cam: Camera, scene_radius: float,
                    fade_scale: float = 0.22) -> torch.Tensor | None:
    """相机附近的球形淡出因子，逐高斯。

    与 WebGL 查看器用同一套参数（viewer.html 里 fadeEnd = sceneRadius * 0.22、
    start = fadeEnd * 0.15），这样训练预览图和实际浏览时看到的一致。

    作用：相机贴近物体时，糊在镜头前的那批高斯会整片挡住画面。它们对成像没有
    贡献（本来就该被视角覆盖掉），但因为高度交叠会吃掉 K 个名额、把真正该显示的
    远处内容挤掉。按距离平滑压掉它们的透明度，既治糊屏也腾出 K。

    注意用 smoothstep 而非硬阈值：硬裁剪会让高斯越过阈值那一帧突然闪一下。
    """
    if fade_scale <= 0:
        return None
    # 相机的 R/T 是 float64，不显式对齐 dtype 的话这个因子会把整条链路带成 double，
    # 官方 CUDA 核只吃 float32，会直接报 "expected scalar type Float but found Double"。
    c = cam.camera_center.to(device=means.device, dtype=means.dtype)
    d = torch.linalg.norm(means - c, dim=-1)
    end = max(scene_radius * fade_scale, 1e-6)
    start = end * 0.15
    t = ((d - start) / max(end - start, 1e-9)).clamp(0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).to(means.dtype)      # smoothstep


def rasterize(
    proj: ProjectedGaussians,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    cam: Camera,
    background: torch.Tensor,
    K: int = DEFAULT_K,
    opacity_scale: torch.Tensor | None = None,
) -> RenderOutput:
    """前向 alpha 合成。proj 由 project_gaussians 得到。

    opacity_scale 是逐高斯的透明度缩放（已激活空间，见 near_fade_scale），
    用于相机附近淡出；None 表示不缩放。
    """
    device, dtype = colors.device, colors.dtype
    W, H = cam.width, cam.height
    num_pixels = W * H

    frags = _build_fragments(proj, cam, device)
    if frags is None:
        img = background.reshape(3, 1, 1).expand(3, H, W).clone()
        return RenderOutput(image=img, num_fragments=0, num_kept=0)

    gids, pixel_ids = frags
    total = gids.shape[0]

    depth = proj.depth[gids].detach()

    # 两次稳定排序实现 (像素, 深度) 字典序：
    # 先按深度升序，再按像素做稳定排序 —— 组内自然保持深度升序，且不需要量化。
    #
    # 这里刻意不用「像素<<24 | 量化深度」的单次复合整数键：量化步长会让深度
    # 相差极小的高斯在任何微扰下反复交换顺序（实测会让 z 方向的数值梯度随
    # eps 反比发散），梯度检验和训练都会因此抖动。
    by_depth = torch.argsort(depth, stable=True)
    order = by_depth[torch.argsort(pixel_ids[by_depth], stable=True)]
    pixel_sorted = pixel_ids[order]
    gids_sorted = gids[order]

    # 每个片元在所属像素内的序号（0 表示最近的）
    slot = torch.arange(total, device=device) - torch.searchsorted(pixel_sorted, pixel_sorted)
    keep = slot < K
    kept = int(keep.sum().item())

    sparse = torch.zeros(num_pixels, K, dtype=torch.long, device=device)
    sparse[pixel_sorted[keep], slot[keep]] = gids_sorted[keep] + 1

    valid_slot = sparse > 0
    gidx = (sparse - 1).clamp(min=0)                 # [P, K]

    # 空槽位必须分散到不同索引，不能全指向 0。
    # 它们在 forward 里被 valid_slot 掩成 alpha=0，梯度恒为 0，指向哪里都无害；
    # 但反向的 scatter-add 仍会为它们执行原子加。若全部指向索引 0，
    # 第 0 个高斯要承受几十万次原子加，实测比分散索引慢 116 倍
    # （634ms -> 5.6ms），是整个训练步最大的隐藏开销。
    gidx = torch.where(valid_slot, gidx, _dummy_indices(gidx.shape, opacities.shape[0], device))

    xs = torch.arange(W, device=device, dtype=dtype).reshape(1, W).expand(H, W).reshape(-1)
    ys = torch.arange(H, device=device, dtype=dtype).reshape(H, 1).expand(H, W).reshape(-1)
    dx = xs.unsqueeze(1) - proj.px[gidx]
    dy = ys.unsqueeze(1) - proj.py[gidx]

    conic = proj.conic[gidx]
    power = -0.5 * (conic[..., 0] * dx * dx + conic[..., 2] * dy * dy) - conic[..., 1] * dx * dy
    opa = torch.sigmoid(opacities[gidx][..., 0])
    if opacity_scale is not None:
        opa = opa * opacity_scale[gidx]
    alpha = opa * torch.exp(torch.clamp(power, max=0.0))
    alpha = alpha * valid_slot.to(dtype)

    one_minus = 1.0 - alpha
    trans = torch.cumprod(one_minus, dim=1)
    # exclusive 前缀积：第 i 个高斯的透射率只乘它前面的
    trans_excl = torch.cat([torch.ones_like(trans[:, :1]), trans[:, :-1]], dim=1)

    weights = alpha * trans_excl
    rgb = (weights.unsqueeze(-1) * colors[gidx]).sum(dim=1)
    rgb = rgb + trans[:, -1:].clamp(min=0.0) * background.reshape(1, 3)

    return RenderOutput(
        image=rgb.reshape(H, W, 3).permute(2, 0, 1).contiguous(),
        num_fragments=total,
        num_kept=kept,
    )


def render_gaussians(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    shs: torch.Tensor,
    cam: Camera,
    background: torch.Tensor,
    sh_degree: int = 3,
    K: int = DEFAULT_K,
    scene_radius: float = 0.0,
    fade_scale: float = 0.0,
) -> RenderOutput:
    """完整前向：球谐求色 -> 投影 -> 光栅化。

    scene_radius + fade_scale 给定时启用相机附近的球形淡出（与查看器同参数）。
    """
    colors = compute_colors(means, shs, cam, sh_degree)
    cov3d = build_cov3d(quats, scales)
    proj = project_gaussians(means, cov3d, cam)
    scale = near_fade_scale(means, cam, scene_radius, fade_scale) if fade_scale > 0 else None
    return rasterize(proj, colors, opacities, cam, background, K=K, opacity_scale=scale)


# --------------------------------------------------------------------------
# 朴素参考实现：逐高斯循环，慢得没法用，但逻辑直白到不可能写错。
# 仅用于验证上面的向量化实现。
# --------------------------------------------------------------------------
def render_naive(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    cam: Camera,
    background: torch.Tensor,
    clip_to_bbox: bool = True,
) -> torch.Tensor:
    """不做分桶、不做截断的参考实现：对每个高斯算 alpha，按深度从近到远合成。

    clip_to_bbox=True（默认）时使用与向量化实现完全相同的整数包围盒，
    因此两者之间的差异只可能来自「分桶 / 排序 / 合成」逻辑。
    clip_to_bbox=False 时保留高斯尾部，用于量化 3-sigma 裁剪本身的误差。
    """
    device, dtype = means.device, means.dtype
    H, W = cam.height, cam.width

    cov3d = build_cov3d(quats, scales)
    proj = project_gaussians(means, cov3d, cam)

    xs = torch.arange(W, device=device, dtype=dtype).reshape(1, W).expand(H, W)
    ys = torch.arange(H, device=device, dtype=dtype).reshape(H, 1).expand(H, W)

    # 注意：背景必须在最后乘上「总透射率」，否则被完全遮挡的像素会多算一层背景。
    img = torch.zeros(3, H, W, device=device, dtype=dtype)
    trans = torch.ones(H, W, device=device, dtype=dtype)

    if clip_to_bbox:
        x0, x1, y0, y1 = compute_bboxes(proj, cam)

    order = torch.argsort(proj.depth)
    for i in order.tolist():
        if not bool(proj.valid[i]):
            continue
        a, b, c = proj.conic[i]
        dx = xs - proj.px[i]
        dy = ys - proj.py[i]
        power = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy
        alpha = torch.sigmoid(opacities[i, 0]) * torch.exp(torch.clamp(power, max=0.0))
        if clip_to_bbox:
            inside = (xs >= x0[i]) & (xs <= x1[i]) & (ys >= y0[i]) & (ys <= y1[i])
            alpha = alpha * inside.to(alpha.dtype)
        img = img + (trans * alpha).unsqueeze(0) * colors[i].reshape(3, 1, 1)
        trans = trans * (1.0 - alpha)

    return img + trans.unsqueeze(0) * background.reshape(3, 1, 1)
