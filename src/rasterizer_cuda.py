"""官方 diff-gaussian-rasterization 的后端包装。

本项目的核心光栅化器是 src/rasterizer.py 里的纯 PyTorch 实现（自己写的，
用于理解算法、并用它做正确性对照）。但纯 PyTorch 的稠密 [像素数 x K] 合成有三个
绕不过去的限制：每像素必须截断到 K 个（训练会因此发散）、显存随 K 线性膨胀、
速度比官方慢两个数量级。

这个模块把官方 CUDA 光栅化器接成同一个接口，作为可切换的加速后端。
两边的输入输出已经逐像素比对过：视觉一致，差异只在边缘高频细节。

编译官方扩展的配方（本机 RTX 5060 / sm_120 实测）：
    CUDA_HOME            = .../CUDA/v13.4
    vcvars64 + MSVC 14.44
    NVCC_PREPEND_FLAGS   = -Xcompiler /Zc:preprocessor   (CUDA 13 的 CCCL 强制)
    DISTUTILS_USE_SDK=1, MSSdk=1                         (torch 的环境检查要)
    TORCH_CUDA_ARCH_LIST = 12.0
见 _upstream/build_dgr.bat。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

from .camera import Camera
from .rasterizer import RenderOutput

_ROOT = Path(__file__).resolve().parents[1]
_DGR_ROOT = _ROOT / "_upstream" / "diff-gaussian-rasterization-main"
_GUTILS = _ROOT / "_upstream"


def _load_official():
    if str(_DGR_ROOT) not in sys.path:
        sys.path.insert(0, str(_DGR_ROOT))
    if str(_GUTILS) not in sys.path:
        sys.path.insert(0, str(_GUTILS))
    try:
        from diff_gaussian_rasterization import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )
    except ImportError as e:
        raise ImportError(
            "官方 CUDA 光栅化器未编译或路径不对，请先跑 _upstream/build_dgr.bat；"
            f"期望位置：{_DGR_ROOT}"
        ) from e
    return GaussianRasterizationSettings, GaussianRasterizer


def render_official(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    shs: torch.Tensor,
    cam: Camera,
    background: torch.Tensor,
    sh_degree: int = 3,
    znear: float = 0.01,
    zfar: float = 100.0,
    scale_modifier: float = 1.0,
    means2d: torch.Tensor | None = None,
):
    """与 render_gaussians 同签名，返回 (RenderOutput, means2d, radii)。

    means2d 是官方光栅化器的屏幕空间坐标输入，它的 .grad 由 CUDA 核填充，
    是官方的密度控制信号来源。外部若需要读梯度，传入自己的可导张量即可。
    """
    GaussianRasterizationSettings, GaussianRasterizer = _load_official()
    dev = means.device
    W, H = cam.width, cam.height

    # 官方约定（utils/graphics_utils.py）：
    #   getWorld2View2(R_cam2world, t) 内部做 R.transpose() 得到 world->cam；
    #   Camera 类里再 .transpose(0,1) 才交给光栅化器；
    #   full_proj = world_view_transform @ projection_matrix（两者都已转置）。
    # 本实现的 cam.R 已经是 world->cam，所以这里要传 cam->world（转置回去）。
    from graphics_utils import getProjectionMatrix, getWorld2View2

    R_cw = cam.R.detach().cpu().numpy()
    T_cw = cam.T.detach().cpu().numpy()
    wvt = torch.from_numpy(getWorld2View2(R_cw.T, T_cw)).to(dev).transpose(0, 1).contiguous()
    fov_x = 2 * math.atan(W / (2 * cam.fx))
    fov_y = 2 * math.atan(H / (2 * cam.fy))
    pm = getProjectionMatrix(znear, zfar, fov_x, fov_y).to(dev).transpose(0, 1).contiguous()
    full_proj = (wvt @ pm).contiguous()

    settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=W / (2 * cam.fx),
        tanfovy=H / (2 * cam.fy),
        bg=background.reshape(3).float(),
        scale_modifier=float(scale_modifier),
        viewmatrix=wvt,
        projmatrix=full_proj,
        sh_degree=sh_degree,
        campos=cam.camera_center.detach().float(),
        prefiltered=False,
        debug=False,
    )
    rast = GaussianRasterizer(raster_settings=settings)
    if means2d is None:
        means2d = torch.zeros_like(means)
    img, radii = rast(
        means3D=means.contiguous(),
        means2D=means2d,
        shs=shs.contiguous(),
        colors_precomp=None,
        # 官方核里是 alpha = opacities * exp(power)（forward.cu:343），没有 sigmoid，
        # 它期望的已经是概率值（官方传 pc.get_opacity = sigmoid(_opacity)）。
        # 本实现内部约定传 logit，所以这里必须自己激活一次：
        # 否则初始 opacity=0.1 对应 logit=-2.197，算出的 alpha 为负，
        # 会被 "alpha < 1/255 就跳过" 整条剔除，梯度恒为 0、训练完全冻住。
        opacities=torch.sigmoid(opacities).contiguous(),
        scales=scales.contiguous(),
        rotations=quats.contiguous(),
        cov3D_precomp=None,
    )
    return RenderOutput(image=img, num_fragments=-1, num_kept=-1), means2d, radii


def official_available() -> bool:
    try:
        _load_official()
        return True
    except Exception:
        return False
