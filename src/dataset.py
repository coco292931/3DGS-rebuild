"""数据集加载。

两个数据集类接口完全一致（camera/image/train_indices/points/background/scene_extent），
训练循环不需要关心数据来自合成场景还是真实拍摄：
    SyntheticDataset  —— tools/synth_scene.py 生成，位姿精确已知
    ColmapDataset     —— 真实视频经 pycolmap SfM 求位姿，见 tools/run_sfm.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .camera import Camera


class SyntheticDataset:
    def __init__(self, root: str | Path, device: str | torch.device = "cuda", test_every: int = 8,
                 downscale: int = 1):
        self.root = Path(root)
        self.device = torch.device(device)
        self.downscale = max(int(downscale), 1)

        cam_data = np.load(self.root / "cameras.npz")
        pts = np.load(self.root / "points.npz")
        self.meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))

        self.R = cam_data["R"]           # [V,3,3]
        self.T = cam_data["T"]           # [V,3]
        self.centers = cam_data["centers"]
        self.fx = float(cam_data["fx"]) / self.downscale
        self.fy = float(cam_data["fy"]) / self.downscale
        self.cx = float(cam_data["cx"]) / self.downscale
        self.cy = float(cam_data["cy"]) / self.downscale
        self.full_width = int(cam_data["width"])
        self.full_height = int(cam_data["height"])
        self.width = self.full_width // self.downscale
        self.height = self.full_height // self.downscale

        self.background = torch.tensor(self.meta["background"], dtype=torch.float32, device=self.device)

        paths = sorted((self.root / "images").glob("*.png"))
        if len(paths) != self.R.shape[0]:
            raise ValueError(f"图像数 {len(paths)} 与相机数 {self.R.shape[0]} 不一致")

        imgs = []
        for p in paths:
            arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1)
            if self.downscale > 1:
                t = F.interpolate(t.unsqueeze(0), size=(self.height, self.width),
                                  mode="area").squeeze(0)
            imgs.append(t.to(self.device))
        self.images = torch.stack(imgs)          # [V,3,H,W]

        self.test_indices = list(range(0, len(paths), test_every))
        self.train_indices = [i for i in range(len(paths)) if i not in set(self.test_indices)]

        self.points = torch.from_numpy(pts["points"]).float().to(self.device)
        self.point_colors = torch.from_numpy(pts["colors"]).float().to(self.device)

    def camera(self, index: int) -> Camera:
        return Camera(
            R=torch.from_numpy(self.R[index]).to(self.device),
            T=torch.from_numpy(self.T[index]).to(self.device),
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            width=self.width, height=self.height,
            device=self.device,
        )

    def image(self, index: int) -> torch.Tensor:
        return self.images[index]

    @property
    def scene_extent(self) -> float:
        pts = self.points.detach().cpu().numpy()
        return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))

    def mean_image_psnr(self, indices=None) -> float:
        """瞎猜基线：用其余视角的平均图当预测，衡量「什么都不学」能拿多少分。"""
        from .metrics import psnr

        indices = self.test_indices if indices is None else indices
        train_mean = self.images[self.train_indices].mean(dim=0)
        return float(np.mean([psnr(train_mean, self.images[i]) for i in indices]))
