"""高斯参数模型：参数存储、激活函数、初始化、PLY 存取、自适应密度控制。

参数布局与官方 3DGS 保持一致，便于 PLY 互通：
    位置      xyz          [N, 3]   直接优化
    旋转      rotation     [N, 4]   四元数 (w,x,y,z)，使用时归一化
    缩放      scaling      [N, 3]   log 空间，使用时 exp
    不透明度  opacity      [N, 1]   logit 空间，使用时 sigmoid
    球谐 DC   features_dc  [N, 1, 3]
    球谐高阶  features_rest[N, 15, 3]
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import torch

from .sh import SH_C0, sh_degree_to_rest_count

# 官方默认值
OPACITY_INIT = 0.1                # 初始不透明度（未取 logit 前）
PERCENT_DENSE = 0.01              # 判定「过平滑」还是「欠采样」的尺度阈值系数
DENSIFY_GRAD_THRESHOLD = 2e-4
PRUNE_MIN_OPACITY = 0.005
PRUNE_MAX_SCREEN_RADIUS = 20.0
PRUNE_MAX_WORLD_SCALE = 0.1


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1.0 - x).clamp(min=1e-12))


def _as_leaf(t: torch.Tensor) -> torch.Tensor:
    """转成带梯度的叶子张量。

    优化器只接受叶子张量。torch.cat / 索引 / 任何算子的输出都带 grad_fn，
    直接塞进优化器会报 "can't optimize a non-leaf Tensor"，
    所以每次重建参数张量都必须过一遍 detach。
    """
    return t.detach().clone().requires_grad_(True)


def build_rotation(quats: torch.Tensor) -> torch.Tensor:
    """四元数 -> 旋转矩阵（同一份公式在 rasterizer 里也有一份，这里供模型侧使用）。"""
    q = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = q.unbind(-1)
    rows = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )
    return rows.reshape(-1, 3, 3)


class GaussianModel:
    def __init__(self, sh_degree: int = 3, device: str | torch.device = "cuda"):
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0
        self.device = torch.device(device)

        self._xyz: torch.Tensor | None = None
        self._features_dc: torch.Tensor | None = None
        self._features_rest: torch.Tensor | None = None
        self._scaling: torch.Tensor | None = None
        self._rotation: torch.Tensor | None = None
        self._opacity: torch.Tensor | None = None

        # 训练期统计量
        self.max_radii2D: torch.Tensor | None = None
        self.xyz_gradient_accum: torch.Tensor | None = None
        self.denom: torch.Tensor | None = None
        self.optimizer: torch.optim.Adam | None = None
        self.spatial_lr_scale = 1.0
        self.scene_extent = 1.0

    # ---------------------------------------------------------------- 属性
    @property
    def num_gaussians(self) -> int:
        return 0 if self._xyz is None else self._xyz.shape[0]

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        return self._rotation / self._rotation.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    @property
    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity)

    @property
    def get_features(self) -> torch.Tensor:
        """[N, K, 3] 把 DC 与高阶拼在一起。"""
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    def oneup_sh_degree(self) -> None:
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # ---------------------------------------------------------------- 初始化
    def init_from_points(
        self,
        points: torch.Tensor,
        colors: torch.Tensor | None = None,
        scene_extent: float | None = None,
        init_opacity: float = OPACITY_INIT,
        scale_factor: float = 1.0,
    ) -> None:
        """用点云初始化。

        缩放初值取「到最近邻距离」，这是官方做法：太大会糊、太小会长不出结构。
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points 必须是 [N,3]，收到 {tuple(points.shape)}")
        n = points.shape[0]
        dev = self.device
        pts = points.to(dev, torch.float32)

        if scene_extent is None:
            # 用点云包围盒对角线作为场景尺度。不要用 cdist 的全局最大距离：
            # 那是 O(N^2) 内存，几万个点就会炸。
            pts_np = pts.detach().cpu().numpy()
            scene_extent = float(np.linalg.norm(pts_np.max(axis=0) - pts_np.min(axis=0))) if n > 1 else 1.0
        self.scene_extent = max(float(scene_extent), 1e-6)

        dists = _nearest_neighbor_distances(pts.detach().cpu().numpy())
        dists = np.clip(dists, 1e-6, None) * float(scale_factor)
        # 初始化尺度直接决定「每像素被多少个高斯覆盖」：
        #   片元总数 ∝ 点数 × 半径² ∝ (σA) × (1/σ) —— 与点云密度无关，
        #   只跟这个 scale_factor 成正比。scale_factor 太小会让表面出现空洞，
        #   但 densification 会补回来；太大则每像素片元数爆炸，K 截断开始丢贡献。
        scales = torch.log(torch.from_numpy(dists).to(dev, torch.float32)).unsqueeze(-1).repeat(1, 3)

        if colors is None:
            colors = torch.full((n, 3), 0.5, device=dev)
        else:
            colors = colors.to(dev, torch.float32).clamp(0.0, 1.0)

        # RGB -> SH DC 系数：rgb = C0 * sh + 0.5
        sh_dc = ((colors - 0.5) / SH_C0).unsqueeze(1)

        self._xyz = pts.clone().requires_grad_(True)
        self._features_dc = sh_dc.contiguous().requires_grad_(True)
        self._features_rest = torch.zeros(
            n, sh_degree_to_rest_count(self.max_sh_degree), 3, device=dev
        ).requires_grad_(True)
        self._scaling = scales.contiguous().requires_grad_(True)
        self._rotation = torch.zeros(n, 4, device=dev)
        self._rotation[:, 0] = 1.0
        self._rotation = self._rotation.requires_grad_(True)
        # 官方初始化的是 inverse_sigmoid(0.1)，直接把 0.1 当 logit 用会让初始不透明度
        # 变成 sigmoid(0.1)=0.525，画面糊成一片
        init_logit = float(inverse_sigmoid(torch.tensor(float(init_opacity))))
        self._opacity = torch.full((n, 1), init_logit, device=dev).requires_grad_(True)

        self.max_radii2D = torch.zeros(n, device=dev)
        self.xyz_gradient_accum = torch.zeros(n, 1, device=dev)
        self.denom = torch.zeros(n, 1, device=dev)

    # ---------------------------------------------------------------- 优化器
    def training_setup(
        self,
        lr_means: float = 1.6e-4,
        lr_features: float = 2.5e-3,
        lr_opacity: float = 5e-2,
        lr_scaling: float = 5e-3,
        lr_rotation: float = 1e-3,
    ) -> torch.optim.Adam:
        params = [
            {"params": [self._xyz], "lr": lr_means * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._features_dc], "lr": lr_features, "name": "f_dc"},
            {"params": [self._features_rest], "lr": lr_features / 20.0, "name": "f_rest"},
            {"params": [self._opacity], "lr": lr_opacity, "name": "opacity"},
            {"params": [self._scaling], "lr": lr_scaling, "name": "scaling"},
            {"params": [self._rotation], "lr": lr_rotation, "name": "rotation"},
        ]
        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        return self.optimizer

    def update_learning_rate(self, iteration: int, lr_means: float = 1.6e-4) -> None:
        """位置学习率按场景尺度指数衰减，官方做法。"""
        if self.optimizer is None:
            return
        for group in self.optimizer.param_groups:
            if group["name"] == "xyz":
                group["lr"] = lr_means * self.spatial_lr_scale * (0.01 ** (iteration / 30000.0))

    # ---------------------------------------------------------------- 密度控制
    def add_densification_stats(self, grad_2d: torch.Tensor, visible: torch.Tensor) -> None:
        """累积屏幕空间位置梯度（用于判断哪里表达不足）。"""
        grad = grad_2d.detach().norm(dim=-1, keepdim=True)
        self.xyz_gradient_accum[visible] += grad[visible]
        self.denom[visible] += 1.0

    def _cat_tensors(self, new_xyz, new_features_dc, new_features_rest, new_scaling, new_rotation, new_opacity):
        self._xyz = _as_leaf(torch.cat([self._xyz, new_xyz], dim=0))
        self._features_dc = _as_leaf(torch.cat([self._features_dc, new_features_dc], dim=0))
        self._features_rest = _as_leaf(torch.cat([self._features_rest, new_features_rest], dim=0))
        self._scaling = _as_leaf(torch.cat([self._scaling, new_scaling], dim=0))
        self._rotation = _as_leaf(torch.cat([self._rotation, new_rotation], dim=0))
        self._opacity = _as_leaf(torch.cat([self._opacity, new_opacity], dim=0))

        # 训练期统计量必须跟着长，否则下一次剪枝的掩码长度会和高斯数量对不上
        k = new_xyz.shape[0]
        if self.max_radii2D is not None:
            self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros(k, device=self.device)])
        if self.xyz_gradient_accum is not None:
            self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, torch.zeros(k, 1, device=self.device)])
        if self.denom is not None:
            self.denom = torch.cat([self.denom, torch.zeros(k, 1, device=self.device)])

    def _prune_points(self, mask: torch.Tensor) -> None:
        """mask 为 True 表示保留。"""
        self._xyz = _as_leaf(self._xyz[mask])
        self._features_dc = _as_leaf(self._features_dc[mask])
        self._features_rest = _as_leaf(self._features_rest[mask])
        self._scaling = _as_leaf(self._scaling[mask])
        self._rotation = _as_leaf(self._rotation[mask])
        self._opacity = _as_leaf(self._opacity[mask])
        if self.max_radii2D is not None:
            self.max_radii2D = self.max_radii2D[mask]
        if self.xyz_gradient_accum is not None:
            self.xyz_gradient_accum = self.xyz_gradient_accum[mask]
        if self.denom is not None:
            self.denom = self.denom[mask]

    def _reset_optimizer(self) -> None:
        self.training_setup()

    def densify_and_clone(self, grads: torch.Tensor, grad_threshold: float) -> None:
        """梯度大 + 尺度小 -> 欠采样，复制一份。"""
        selected = (grads.squeeze(-1) >= grad_threshold) & (
            self.get_scaling.max(dim=1).values <= PERCENT_DENSE * self.scene_extent
        )
        if not bool(selected.any()):
            return self.num_gaussians
        self._cat_tensors(
            self._xyz[selected].detach(),
            self._features_dc[selected].detach(),
            self._features_rest[selected].detach(),
            self._scaling[selected].detach(),
            self._rotation[selected].detach(),
            self._opacity[selected].detach(),
        )
        return self.num_gaussians

    def densify_and_split(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        n_split: int = 2,
        seed: int | None = None,
        extra_mask: torch.Tensor | None = None,
    ) -> None:
        """梯度大 + 尺度大 -> 过平滑，一分为二。

        新位置在原始高斯的主轴方向上按标准差采样，尺度缩小 0.8*2 倍。

        extra_mask 是官方的盲区补丁：官方只按「梯度大」触发 split，所以一个
        「变大但梯度小」（已经拟合得不错）的高斯永远不会被拆。官方 tile 渲染没有
        K 上限，无所谓；本实现每像素只保留 K 个，这类大高斯会持续推高片元数，
        最终把训练推向发散。所以额外把「屏幕半径超标」的高斯也拆掉。
        注意必须是 split 而不是 prune：剪掉会让近处地面点成片消失（实测过，地面整个没）。
        """
        n_init = self.num_gaussians
        selected = (grads.squeeze(-1) >= grad_threshold) & (
            self.get_scaling.max(dim=1).values > PERCENT_DENSE * self.scene_extent
        )
        if extra_mask is not None:
            selected = selected | extra_mask
        if not bool(selected.any()):
            return n_init

        stds = self.get_scaling[selected].repeat(n_split, 1)
        means = torch.zeros_like(stds)
        if seed is not None:
            gen = torch.Generator(device=stds.device).manual_seed(seed)
            samples = torch.normal(mean=means, std=stds, generator=gen)
        else:
            samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected]).repeat(n_split, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected].repeat(n_split, 1)
        new_scaling = torch.log(self.get_scaling[selected].repeat(n_split, 1) / (0.8 * n_split))
        new_rotation = self._rotation[selected].repeat(n_split, 1)
        new_features_dc = self._features_dc[selected].repeat(n_split, 1, 1)
        new_features_rest = self._features_rest[selected].repeat(n_split, 1, 1)
        new_opacity = self._opacity[selected].repeat(n_split, 1)

        keep = ~selected
        self._prune_points(keep)
        self._cat_tensors(
            new_xyz.detach(), new_features_dc.detach(), new_features_rest.detach(),
            new_scaling.detach(), new_rotation.detach(), new_opacity.detach(),
        )
        assert self.num_gaussians == n_init - int(selected.sum()) + int(selected.sum()) * n_split
        return self.num_gaussians

    def prune(self, min_opacity: float = PRUNE_MIN_OPACITY, max_screen_radius: float = PRUNE_MAX_SCREEN_RADIUS) -> dict:
        """剪掉透明的高斯，以及被相机怼到屏幕上、尺度失控的高斯。

        返回各条件的命中数，便于诊断「一次剪掉一大半」到底是哪个阈值在起作用。
        """
        low_opacity = (self.get_opacity < min_opacity).squeeze(-1)
        big_screen = self.max_radii2D > max_screen_radius
        big_world = self.get_scaling.max(dim=1).values > PRUNE_MAX_WORLD_SCALE * self.scene_extent
        prune_mask = low_opacity | big_screen | big_world
        stats = {
            "low_opacity": int(low_opacity.sum().item()),
            "big_screen": int(big_screen.sum().item()),
            "big_world": int(big_world.sum().item()),
            "pruned": int(prune_mask.sum().item()),
            "total": self.num_gaussians,
        }
        if bool(prune_mask.any()):
            self._prune_points(~prune_mask)
        return stats

    def reset_opacity(self, max_opacity: float = 0.01) -> None:
        """把所有高斯的不透明度压回去，逼着「又大又透明」的垃圾重新竞争。"""
        new_opacity = torch.min(self.get_opacity, torch.full_like(self.get_opacity, max_opacity))
        self._opacity = _as_leaf(inverse_sigmoid(new_opacity.clamp(1e-6, 1 - 1e-6)))

    def densify_and_prune(
        self,
        max_grad: float,
        min_opacity: float,
        max_gaussians: int,
        seed: int | None = None,
        max_screen_radius: float = PRUNE_MAX_SCREEN_RADIUS,
        split_radius_px: float = 0.0,
        max_split_frac: float = 0.25,
    ) -> dict:
        """一轮完整的增密 + 剪枝，返回统计信息。"""
        before = self.num_gaussians
        grads = self.xyz_gradient_accum / self.denom.clamp(min=1.0)
        grads[grads.isnan()] = 0.0

        # 屏幕半径超标的高斯：无条件拆（不看梯度）。限制每轮比例，防止一轮拆爆。
        extra = None
        if split_radius_px and split_radius_px > 0 and self.max_radii2D is not None:
            extra = self.max_radii2D > split_radius_px
            limit = int(self.num_gaussians * max_split_frac)
            if limit > 0 and int(extra.sum()) > limit:
                top = torch.topk(self.max_radii2D, limit).indices
                extra = torch.zeros_like(extra)
                extra[top] = True

        after_clone = self.densify_and_clone(grads, max_grad)
        # clone 之后高斯数变了，梯度向量要补零对齐（对应官方的 padded_grad）。
        # 补出来的零梯度意味着新克隆的点本轮不参与 split。
        if self.num_gaussians != grads.shape[0]:
            pad = torch.zeros(self.num_gaussians - grads.shape[0], grads.shape[1],
                              device=grads.device, dtype=grads.dtype)
            grads = torch.cat([grads, pad], dim=0)
            if extra is not None:
                extra = torch.cat([extra, torch.zeros(pad.shape[0], dtype=torch.bool, device=grads.device)])
        after_split = self.densify_and_split(grads, max_grad, seed=seed, extra_mask=extra)
        prune_stats = self.prune(min_opacity=min_opacity, max_screen_radius=max_screen_radius)

        # 显存保险：本实现用稠密合成张量，必须给高斯数量设上限
        if self.num_gaussians > max_gaussians:
            keep = torch.rand(self.num_gaussians, device=self.device) < (max_gaussians / self.num_gaussians)
            self._prune_points(keep)

        self.xyz_gradient_accum = torch.zeros(self.num_gaussians, 1, device=self.device)
        self.denom = torch.zeros(self.num_gaussians, 1, device=self.device)
        self.max_radii2D = torch.zeros(self.num_gaussians, device=self.device)
        self._reset_optimizer()

        return {"before": before, "after": self.num_gaussians, "clone": after_clone, "split": after_split,
                "prune": prune_stats,
                "forced_split": int(extra.sum()) if extra is not None else 0}

    # ---------------------------------------------------------------- PLY
    def save_ply(self, path: str | Path) -> None:
        """写出官方兼容的 3DGS PLY（binary_little_endian）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = self.num_gaussians

        xyz = self._xyz.detach().cpu().numpy().astype(np.float32)
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(1).cpu().numpy().astype(np.float32)
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(1).cpu().numpy().astype(np.float32)
        opacity = self._opacity.detach().cpu().numpy().astype(np.float32)
        scale = self._scaling.detach().cpu().numpy().astype(np.float32)
        rot = self._rotation.detach().cpu().numpy().astype(np.float32)

        fields = (
            [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
            + [(f"f_dc_{i}", "f4") for i in range(3)]
            + [(f"f_rest_{i}", "f4") for i in range(f_rest.shape[1])]
            + [("opacity", "f4")]
            + [(f"scale_{i}", "f4") for i in range(3)]
            + [(f"rot_{i}", "f4") for i in range(4)]
        )
        data = np.concatenate([xyz, normals, f_dc, f_rest, opacity, scale, rot], axis=1).astype(np.float32)

        header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
        header += [f"property float {name}" for name, _ in fields]
        header += ["end_header"]
        # 原子写：先落临时文件再 rename。模型有十几 MB，直接写目标文件的话，
        # 正在轮询 checkpoint 目录的看板会读到半成品，PLY 解析当场失败。
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            fh.write(("\n".join(header) + "\n").encode("ascii"))
            fh.write(data.tobytes())
        os.replace(tmp, path)

    @classmethod
    def load_ply(cls, path: str | Path, sh_degree: int = 3, device: str | torch.device = "cuda") -> "GaussianModel":
        model = cls(sh_degree=sh_degree, device=device)
        names, data = read_ply_vertex_data(Path(path))
        idx = {name: i for i, name in enumerate(names)}

        def col(name: str) -> np.ndarray:
            if name not in idx:
                raise ValueError(f"{path} 缺少属性 {name}")
            return data[:, idx[name]]

        def cols(prefix: str, count: int) -> np.ndarray:
            return np.stack([col(f"{prefix}{i}") for i in range(count)], axis=1)

        def to_param(arr: np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(arr)).to(device).float().requires_grad_(True)

        n_rest = sum(1 for name in names if name.startswith("f_rest_"))
        expected_rest = 3 * sh_degree_to_rest_count(sh_degree)
        if n_rest != expected_rest:
            raise ValueError(
                f"PLY 的高阶球谐数量 {n_rest} 与 sh_degree={sh_degree} 期望的 {expected_rest} 不符"
            )

        model._xyz = to_param(np.stack([col("x"), col("y"), col("z")], axis=1))
        model._features_dc = to_param(cols("f_dc_", 3)).reshape(-1, 1, 3)
        # 存储是 channel-major（[N,3,K] 展平），读回要还原成 [N,K,3]
        rest = cols("f_rest_", n_rest).reshape(-1, 3, n_rest // 3).transpose(0, 2, 1)
        model._features_rest = to_param(rest)
        model._opacity = to_param(col("opacity").reshape(-1, 1))
        model._scaling = to_param(cols("scale_", 3))
        model._rotation = to_param(cols("rot_", 4))

        n = model._xyz.shape[0]
        model.max_radii2D = torch.zeros(n, device=device)
        model.xyz_gradient_accum = torch.zeros(n, 1, device=device)
        model.denom = torch.zeros(n, 1, device=device)
        pts_np = model._xyz.detach().cpu().numpy()
        model.scene_extent = float(np.linalg.norm(pts_np.max(axis=0) - pts_np.min(axis=0))) if n > 1 else 1.0
        return model


def _nearest_neighbor_distances(points: np.ndarray) -> np.ndarray:
    """到最近邻的距离，用 cKDTree；点数少时退化成暴力计算。"""
    n = points.shape[0]
    if n <= 1:
        return np.ones(max(n, 1), dtype=np.float32) * 0.01
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        d, _ = tree.query(points, k=2)
        return d[:, 1].astype(np.float32)
    except Exception:
        d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        return d.min(axis=1).astype(np.float32)


def read_ply_vertex_data(path: Path) -> tuple[list[str], np.ndarray]:
    """读 binary_little_endian PLY 的顶点属性，返回 (属性名, [N,K] 数组)。"""
    with open(path, "rb") as fh:
        raw = fh.read()

    header_end = raw.find(b"end_header")
    if header_end < 0:
        raise ValueError(f"{path} 不是合法的 PLY（缺少 end_header）")
    header_text = raw[:header_end].decode("ascii", errors="replace")
    body_start = raw.find(b"\n", header_end) + 1

    names: list[str] = []
    count = None
    for line in header_text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
            count = int(parts[2])
        elif len(parts) >= 3 and parts[0] == "property" and parts[1] == "float":
            names.append(parts[2])
    if count is None:
        raise ValueError(f"{path} 缺少 vertex 元素")

    k = len(names)
    data = np.frombuffer(raw, dtype=np.float32, count=count * k, offset=body_start)
    return names, data.reshape(count, k)
