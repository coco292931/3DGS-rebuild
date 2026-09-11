"""真实场景数据集：读取 COLMAP 的 SfM 结果（pycolmap 4.x 格式）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree

from .camera import Camera




def _rotation_to_z(v: np.ndarray) -> np.ndarray:
    """构造一个把方向 v 转到 +z 的最小旋转矩阵。"""
    v = np.asarray(v, dtype=np.float64)
    v = v / max(np.linalg.norm(v), 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(v, z))
    if c > 1.0 - 1e-9:
        return np.eye(3)
    if c < -1.0 + 1e-9:
        return np.diag([1.0, -1.0, -1.0])   # 反向，绕 x 转 180 度（保持右手系）
    axis = np.cross(v, z)
    axis /= np.linalg.norm(axis)
    ang = float(np.arccos(np.clip(c, -1.0, 1.0)))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
class ColmapDataset:
    """与 SyntheticDataset 接口一致，训练循环无需区分数据来源。"""

    def __init__(self, root, device="cuda", test_every=8, downscale=2,
                 max_points=80000, min_track_length=3, max_reprojection_error=1.5,
                 knn_percentile=97.0, align_up=True):
        import pycolmap

        self.root = Path(root)
        self.device = torch.device(device)
        self.downscale = max(int(downscale), 1)

        model_dir = self._find_model(self.root)
        rec = pycolmap.Reconstruction(str(model_dir))
        self.rec = rec

        cam = list(rec.cameras.values())[0]
        self.fx = cam.focal_length_x / self.downscale
        self.fy = cam.focal_length_y / self.downscale
        self.cx = cam.principal_point_x / self.downscale
        self.cy = cam.principal_point_y / self.downscale
        self.width = cam.width // self.downscale
        self.height = cam.height // self.downscale

        image_dir = self.root / "frames"
        entries = []
        for img_id, img in sorted(rec.images.items(), key=lambda kv: kv[1].name):
            pose = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
            R = np.asarray(pose.rotation.matrix(), dtype=np.float64)
            t = np.asarray(pose.translation, dtype=np.float64)
            p = image_dir / Path(img.name).name
            if not p.exists():
                continue
            entries.append((str(p), R, t))

        if not entries:
            raise RuntimeError(f"{image_dir} 下没有找到任何已注册的图像")

        self.R = np.stack([e[1] for e in entries])
        self.T = np.stack([e[2] for e in entries])
        self.centers = np.stack([-e[1].T @ e[2] for e in entries])
        self.paths = [e[0] for e in entries]

        # 图像常驻 CPU 内存（202 张 640x360 约 140MB），用时再搬上显存
        imgs = []
        for p in self.paths:
            arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1)
            if self.downscale > 1:
                t = F.interpolate(t.unsqueeze(0), size=(self.height, self.width), mode="area").squeeze(0)
            imgs.append(t)
        self.images = torch.stack(imgs)

        # 背景色：取所有图像四周边框像素的中位数，比硬编码黑/白更贴近实拍
        edge = torch.cat([
            self.images[:, :, :4, :].reshape(len(imgs), 3, -1),
            self.images[:, :, -4:, :].reshape(len(imgs), 3, -1),
            self.images[:, :, :, :4].reshape(len(imgs), 3, -1),
            self.images[:, :, :, -4:].reshape(len(imgs), 3, -1),
        ], dim=2)
        self.background = edge.median(dim=2).values.median(dim=0).values.to(self.device)

        # 稀疏点云：过滤短轨迹与离群点后作为高斯初始化
        centers = self.centers
        med = np.median(centers, axis=0)
        radius = float(np.median(np.linalg.norm(centers - med, axis=1)))
        # 三重过滤。单靠「到中心距离」不够：实测 KNN 距离的 p99 是 p50 的 19 倍，
        # 空间上稀疏散布的那些点才是真正会把高斯甩到远处的元凶。
        raw, cols_raw = [], []
        n_raw = 0
        for p in rec.points3D.values():
            n_raw += 1
            if p.track.length() < min_track_length:
                continue
            raw.append([*p.xyz, p.error])
            cols_raw.append(np.asarray(p.color, dtype=np.float64) / 255.0)
        raw = np.asarray(raw) if raw else np.zeros((0, 4))
        cols_all = np.asarray(cols_raw) if cols_raw else np.zeros((0, 3))

        keep = np.ones(len(raw), dtype=bool)
        if len(raw):
            keep &= raw[:, 3] < max_reprojection_error
            keep &= np.linalg.norm(raw[:, :3] - med, axis=1) < 3.0 * radius
            # 空间离群：到 5 近邻的平均距离超过 p97 的点基本都是漂浮外点
            sample = np.random.default_rng(0).choice(len(raw), min(4000, len(raw)), replace=False)
            tree = cKDTree(raw[:, :3])
            kd, _ = tree.query(raw[:, :3][sample], k=6)
            knn_all = kd[:, 1:].mean(axis=1)
            thr = float(np.percentile(knn_all, knn_percentile))
            kd_full, _ = tree.query(raw[:, :3], k=6)
            keep &= kd_full[:, 1:].mean(axis=1) <= thr
        pts = raw[keep][:, :3] if len(raw) else np.zeros((0, 3))
        cols = cols_all[keep] if len(raw) else np.zeros((0, 3))
        self.filter_stats = {
            "raw_points": n_raw, "kept_points": int(pts.shape[0]),
            "camera_radius": radius, "prune_radius": 3.0 * radius,
            "knn_threshold": float(thr) if len(raw) else 0.0,
            "dropped_by_error": int((raw[:, 3] >= max_reprojection_error).sum()) if len(raw) else 0,
            "dropped_by_knn": int((~keep).sum()) if len(raw) else 0,
        }
        if pts.shape[0] > max_points:
            sel = np.random.default_rng(0).choice(pts.shape[0], max_points, replace=False)
            pts, cols = pts[sel], cols[sel]
        # SfM 的世界坐标系由前两帧决定，和「z 朝上」没有任何约定关系。
        # 实测这个场景里「图像下方」平均指向世界 +z（+0.46），也就是整个场景是倒的。
        # 这里用点云主平面的法线定 up（桌面占多数点），再用相机在哪一侧定符号，
        # 把整个场景旋到 z 朝上。R' = R @ R_align^T 保证 p_cam = R' p'_world + t 仍成立。
        if align_up and len(self.R) > 3:
            # up 用「相机图像上方的平均方向」定，不要用点云主平面法线：
            # 后者会被离群点带偏（本场景点云 z 范围 -3.6..8.8，均值中心被拉到高处，
            # 结果相机反而落在点云下方，符号判反、场景照样是倒的）。
            # 相机 y 轴是图像「下」方向，取负得到「上」，人在拍摄时不会倒着拿手机，
            # 所以这个平均方向是可靠的 up 估计。
            y_axes = np.stack([self.R[i][1] for i in range(len(self.R))])
            up_avg = -y_axes.mean(axis=0)
            R_align = _rotation_to_z(up_avg)
            # 新世界坐标 p' = R_align p，即 p = R_align^T p'，代回 p_cam = R p + t 得
            #     p_cam = (R @ R_align^T) p' + t   =>   R_new = R_old @ R_align^T
            # 不要写成 R_align @ R_old：那是把旋转作用在了相机轴上，
            # 结果场景根本不会转正（实测对完还是倒的）。
            self.R = self.R @ R_align.T
            self.centers = self.centers @ R_align.T
            # 点云必须跟着一起转！只转相机的话，初始高斯还留在旧坐标系里，
            # 和相机对不上，投影到屏幕上的片元数会从 400 万掉到 11 万，训练直接废掉。
            if len(pts):
                pts = pts @ R_align.T
            self.align_matrix = R_align
            # 自检：对齐后相机各轴都要一致地转过去
            check = float((-self.R[:, 1, :].mean(axis=0))[2])
            raw_consistency = float(np.linalg.norm(y_axes.mean(axis=0)))
            self.align_info = {
                "up_avg_raw": up_avg.tolist(),
                "up_z_after": check,
                "rotated_deg": float(np.degrees(np.arccos(np.clip((np.trace(R_align) - 1) / 2, -1, 1)))),
                "camera_consistency": raw_consistency,
            }
        else:
            self.align_matrix = np.eye(3)
            self.align_info = {"rotated_deg": 0.0, "up_z_after": float("nan")}

        self.points = torch.from_numpy(pts).float().to(self.device)
        self.point_colors = torch.from_numpy(cols).float().to(self.device)

        n = len(self.paths)
        self.test_indices = list(range(0, n, test_every))
        self.train_indices = [i for i in range(n) if i not in set(self.test_indices)]

    def world_to_ply_space(self, pts: np.ndarray) -> np.ndarray:
        """把模型里保存的坐标转回训练时使用的（已对齐）坐标系。"""
        return pts @ self.align_matrix.T

    @staticmethod
    def _find_model(root: Path) -> Path:
        """挑注册图像最多的模型。

        incremental_mapping 可能输出多个模型（本场景就输出了两个：81 图和 142 图，
        相机绕圈途中因光照突变断过一次）。写死 sparse/0 会拿到小的那个，
        白白丢掉一半视角——实测只用 81 帧时 PSNR 只有 16.8 dB。
        """
        import pycolmap

        sparse = root / "sparse"
        cands = []
        if sparse.exists():
            cands = [d for d in sorted(sparse.iterdir())
                     if d.is_dir() and ((d / "cameras.bin").exists() or (d / "cameras.txt").exists())]
        if not cands:
            cands = [d for d in (root, sparse) if (d / "cameras.bin").exists() or (d / "cameras.txt").exists()]
        if not cands:
            raise FileNotFoundError(f"{root} 下找不到 COLMAP 模型（cameras.bin）")

        best, best_n = cands[0], -1
        for d in cands:
            try:
                n = pycolmap.Reconstruction(str(d)).num_reg_images()
            except Exception:
                continue
            if n > best_n:
                best, best_n = d, n
        return best

    def camera(self, index: int) -> Camera:
        return Camera(
            R=torch.from_numpy(self.R[index]).to(self.device),
            T=torch.from_numpy(self.T[index]).to(self.device),
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            width=self.width, height=self.height, device=self.device,
        )

    def image(self, index: int) -> torch.Tensor:
        return self.images[index].to(self.device)

    @property
    def scene_extent(self) -> float:
        c = self.centers
        return float(np.linalg.norm(c - c.mean(axis=0), axis=1).max())

    def mean_image_psnr(self, indices=None) -> float:
        """瞎猜基线：拿训练集的平均图当预测，衡量「什么都不学」能拿多少分。"""
        from .metrics import psnr

        indices = self.test_indices if indices is None else indices
        train_mean = self.images[self.train_indices].mean(dim=0).to(self.device)
        return float(np.mean([psnr(train_mean, self.image(i)) for i in indices]))