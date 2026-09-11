"""程序化合成场景 + 独立的三角形软件渲染器，用于生成 3DGS 的训练数据。

为什么不用高斯自己渲染 GT：
    那样就是「用高斯拟合高斯」，PSNR 再高也不能证明实现正确。
    这里的 GT 由三角形光栅化（z-buffer + 透视正确插值 + Lambert 光照）产生，
    和高斯管线的数学完全独立。3DGS 去拟合它，指标才有验证意义。

坐标系：z 轴向上。

用法：
    python tools/synth_scene.py --out data/synthetic --n-views 80
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 材质图案类型
PATTERN_SOLID = 0
PATTERN_CHECKER = 1
PATTERN_STRIPES = 2


class Mesh:
    """三角形 soup：世界坐标顶点 + 顶点法线 + 材质。"""

    def __init__(self):
        self.verts: list[np.ndarray] = []      # 每个元素 (3,3)：三个顶点
        self.normals: list[np.ndarray] = []    # 每个元素 (3,3)：三个顶点法线
        self.colors: list[np.ndarray] = []     # 每个元素 (3,)：基色
        self.patterns: list[int] = []

    def add_triangle(self, v0, v1, v2, color, pattern=PATTERN_SOLID, normals=None):
        v0, v1, v2 = (np.asarray(v, dtype=np.float64) for v in (v0, v1, v2))
        if normals is None:
            n = np.cross(v1 - v0, v2 - v0)
            n = n / max(np.linalg.norm(n), 1e-12)
            normals = np.stack([n, n, n])
        self.verts.append(np.stack([v0, v1, v2]))
        self.normals.append(np.asarray(normals, dtype=np.float64))
        self.colors.append(np.asarray(color, dtype=np.float64))
        self.patterns.append(int(pattern))

    def add_quad(self, v0, v1, v2, v3, color, pattern=PATTERN_SOLID, normals=None):
        self.add_triangle(v0, v1, v2, color, pattern, normals)
        self.add_triangle(v0, v2, v3, color, pattern, normals)

    def add_box(self, center, size, color, rot_z=0.0):
        c = np.asarray(center, dtype=np.float64)
        sx, sy, sz = np.asarray(size, dtype=np.float64) / 2.0
        corners = np.array([
            [-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
            [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz],
        ])
        cz, sz_ = np.cos(rot_z), np.sin(rot_z)
        R = np.array([[cz, -sz_, 0], [sz_, cz, 0], [0, 0, 1]])
        pts = corners @ R.T + c
        faces = [
            (0, 3, 2, 1), (4, 5, 6, 7),
            (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
        ]
        for a, b, cc, d in faces:
            self.add_quad(pts[a], pts[b], pts[cc], pts[d], color)

    def add_sphere(self, center, radius, color, n_lat=12, n_lon=18, pattern=PATTERN_STRIPES):
        c = np.asarray(center, dtype=np.float64)
        lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat + 1)
        lon = np.linspace(0, 2 * np.pi, n_lon + 1)
        grid = np.zeros((n_lat + 1, n_lon + 1, 3))
        for i, la in enumerate(lat):
            for j, lo in enumerate(lon):
                grid[i, j] = [
                    radius * np.cos(la) * np.cos(lo),
                    radius * np.cos(la) * np.sin(lo),
                    radius * np.sin(la),
                ] + c

        def normal_at(p):
            n = p - c
            return n / max(np.linalg.norm(n), 1e-12)

        for i in range(n_lat):
            for j in range(n_lon):
                p00, p01, p10, p11 = grid[i, j], grid[i, j + 1], grid[i + 1, j], grid[i + 1, j + 1]
                self.add_triangle(
                    p00, p10, p11, color, pattern,
                    np.stack([normal_at(p00), normal_at(p10), normal_at(p11)]),
                )
                self.add_triangle(
                    p00, p11, p01, color, pattern,
                    np.stack([normal_at(p00), normal_at(p11), normal_at(p01)]),
                )

    def add_plane(self, origin, du, dv, color, pattern=PATTERN_CHECKER, nu=1, nv=1):
        o = np.asarray(origin, dtype=np.float64)
        du = np.asarray(du, dtype=np.float64)
        dv = np.asarray(dv, dtype=np.float64)
        n = np.cross(du, dv)
        n = n / max(np.linalg.norm(n), 1e-12)
        for i in range(nu):
            for j in range(nv):
                p00 = o + du * (i / nu) + dv * (j / nv)
                p10 = o + du * ((i + 1) / nu) + dv * (j / nv)
                p11 = o + du * ((i + 1) / nu) + dv * ((j + 1) / nv)
                p01 = o + du * (i / nu) + dv * ((j + 1) / nv)
                ns = np.stack([n, n, n])
                self.add_triangle(p00, p10, p11, color, pattern, ns)
                self.add_triangle(p00, p11, p01, color, pattern, ns)

    def to_arrays(self):
        return (
            np.stack(self.verts),
            np.stack(self.normals),
            np.stack(self.colors),
            np.array(self.patterns, dtype=np.int64),
        )


def build_scene() -> Mesh:
    m = Mesh()
    # 地面：棋盘，法线朝上（注意顶点朝向要让法线指向 +z）
    m.add_plane(origin=[-6, -6, 0], du=[12, 0, 0], dv=[0, 12, 0],
                color=[0.55, 0.55, 0.58], pattern=PATTERN_CHECKER)

    # 三个立方体
    m.add_box(center=[-0.95, 0.55, 0.40], size=[0.80, 0.80, 0.80], color=[0.85, 0.25, 0.22], rot_z=0.35)
    m.add_box(center=[0.85, -0.75, 0.32], size=[0.64, 0.64, 0.64], color=[0.25, 0.65, 0.35], rot_z=-0.5)
    m.add_box(center=[0.35, 1.15, 0.25], size=[0.50, 0.50, 0.50], color=[0.20, 0.35, 0.80], rot_z=0.9)

    # 一个球
    m.add_sphere(center=[-0.15, -0.25, 0.42], radius=0.42, color=[0.92, 0.82, 0.25])

    return m


def _shade(world, normal, base_color, pattern, light_dir, ambient=0.30, diffuse=0.75):
    """Lambert 光照 + 图案。world/normal/base_color 形状均为 [..., 3]。"""
    lam = np.clip(-(normal * light_dir).sum(axis=-1), 0.0, None)
    intensity = ambient + diffuse * lam

    # pattern 可能是标量（逐三角形渲染）或数组（批量采样点），统一用 where 广播
    pat = np.asarray(pattern)
    cell = np.floor(world[..., 0] / 0.5) + np.floor(world[..., 1] / 0.5)
    band = np.floor(world[..., 2] * 6.0 + 0.5 * world[..., 0] * 3.0)
    checker = np.where((cell % 2) == 0, 1.0, 0.62)
    stripes = np.where((band % 2) == 0, 1.0, 0.70)
    tint = np.where(pat == PATTERN_CHECKER, checker, np.where(pat == PATTERN_STRIPES, stripes, 1.0))

    rgb = base_color * tint[..., None] * intensity[..., None]
    return np.clip(rgb, 0.0, 1.0)


def render_mesh(verts, normals, colors, patterns, cam, width, height, light_dir, bg):
    """透视正确的三角形光栅化：z-buffer + 屏幕空间重心插值。

    返回 [H,W,3] 的 float 图像。
    """
    R, T, fx, fy, cx, cy = cam
    n_faces = verts.shape[0]

    cam_v = verts @ R.T + T                      # [F,3,3]
    z = cam_v[..., 2]
    face_ok = (z > 0.05).all(axis=1)

    u = fx * cam_v[..., 0] / np.maximum(z, 1e-9) + cx
    v = fy * cam_v[..., 1] / np.maximum(z, 1e-9) + cy

    img = np.tile(np.asarray(bg, dtype=np.float64).reshape(1, 1, 3), (height, width, 1))
    zbuf = np.full((height, width), np.inf)

    for i in range(n_faces):
        if not face_ok[i]:
            continue
        ux0 = max(int(np.floor(u[i].min())), 0)
        ux1 = min(int(np.ceil(u[i].max())), width - 1)
        uy0 = max(int(np.floor(v[i].min())), 0)
        uy1 = min(int(np.ceil(v[i].max())), height - 1)
        if ux1 < ux0 or uy1 < uy0:
            continue

        gx, gy = np.meshgrid(np.arange(ux0, ux1 + 1) + 0.5, np.arange(uy0, uy1 + 1) + 0.5)

        ax, ay = u[i, 0], v[i, 0]
        bx, by = u[i, 1], v[i, 1]
        ddx, ddy = u[i, 2], v[i, 2]
        area = (bx - ax) * (ddy - ay) - (by - ay) * (ddx - ax)
        if abs(area) < 1e-12:
            continue

        w0 = ((bx - gx) * (ddy - gy) - (by - gy) * (ddx - gx)) / area
        w1 = ((ddx - gx) * (ay - gy) - (ddy - gy) * (ax - gx)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue

        # 透视正确插值：屏幕空间线性插值的是 1/z
        inv_z = w0 / z[i, 0] + w1 / z[i, 1] + w2 / z[i, 2]
        depth = 1.0 / np.maximum(inv_z, 1e-12)
        ww = np.stack([w0 / z[i, 0], w1 / z[i, 1], w2 / z[i, 2]], axis=-1) / np.maximum(inv_z, 1e-12)[..., None]

        world = ww[..., 0:1] * verts[i, 0] + ww[..., 1:2] * verts[i, 1] + ww[..., 2:3] * verts[i, 2]
        nrm = ww[..., 0:1] * normals[i, 0] + ww[..., 1:2] * normals[i, 1] + ww[..., 2:3] * normals[i, 2]
        nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)

        rgb = _shade(world, nrm, colors[i], patterns[i], light_dir)

        sub_z = zbuf[uy0:uy1 + 1, ux0:ux1 + 1]
        take = inside & (depth < sub_z)
        if not take.any():
            continue
        sub_z[take] = depth[take]
        img[uy0:uy1 + 1, ux0:ux1 + 1][take] = rgb[take]

    return img


def sample_points(mesh_arrays, n_points, rng, light_dir, embed_ratio=1.0):
    """按三角形面积加权在表面采样点，颜色用与渲染同一套光照+图案（模拟 SfM 点云）。"""
    verts, normals, colors, patterns = mesh_arrays
    a = np.cross(verts[:, 1] - verts[:, 0], verts[:, 2] - verts[:, 0])
    areas = 0.5 * np.linalg.norm(a, axis=1)
    probs = areas / areas.sum()

    pick = rng.choice(verts.shape[0], size=n_points, p=probs)
    r1 = rng.random((n_points, 1))
    r2 = rng.random((n_points, 1))
    flip = r1 + r2 > 1.0
    r1 = np.where(flip, 1.0 - r1, r1)
    r2 = np.where(flip, 1.0 - r2, r2)
    pts = (
        verts[pick, 0] * (1 - r1 - r2)
        + verts[pick, 1] * r1
        + verts[pick, 2] * r2
    )
    nrm = normals[pick, 0]
    pt_colors = _shade(pts, nrm, colors[pick], patterns[pick], light_dir)
    # 模拟 SfM 的定位误差
    pts = pts + rng.normal(0.0, 0.004 * embed_ratio, size=pts.shape)
    return pts, pt_colors


def main() -> int:
    ap = argparse.ArgumentParser(description="生成合成场景数据集")
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n-views", type=int, default=80)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--radius", type=float, default=3.6)
    ap.add_argument("--height-cam", type=float, default=1.7)
    ap.add_argument("--n-points", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    mesh = build_scene()
    arrays = mesh.to_arrays()
    verts, normals, colors, patterns = arrays
    print(f"场景：{verts.shape[0]} 个三角形")

    W, H = args.width, args.height
    fov_x = 60.0
    fx = 0.5 * W / np.tan(np.deg2rad(fov_x) / 2.0)
    fy = fx
    cx, cy = W / 2.0, H / 2.0

    light_dir = np.array([0.45, -0.55, -0.70])
    light_dir = light_dir / np.linalg.norm(light_dir)
    bg = np.array([0.62, 0.72, 0.85])           # 天空色

    # 相机绕天顶轴均匀一圈，视线略俯向原点
    cam_centers = []
    cam_R = []
    cam_T = []
    for i in range(args.n_views):
        theta = 2 * np.pi * i / args.n_views
        eye = np.array([args.radius * np.cos(theta), args.radius * np.sin(theta), args.height_cam])
        target = np.array([0.0, 0.0, 0.35])
        eye = eye + np.random.default_rng(args.seed + i).normal(0, 0.02, size=3)  # 轻微的位姿噪声

        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        up_world = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up_world)
        right = right / np.linalg.norm(right)
        down = np.cross(forward, right)
        R = np.stack([right, down, forward])
        T = -R @ eye

        cam_centers.append(eye)
        cam_R.append(R)
        cam_T.append(T)

        img = render_mesh(arrays[0], arrays[1], arrays[2], arrays[3],
                          (R, T, fx, fy, cx, cy), W, H, light_dir, bg)
        from PIL import Image
        Image.fromarray((img * 255.0 + 0.5).astype(np.uint8)).save(out / "images" / f"{i:04d}.png")
        if (i + 1) % 20 == 0:
            print(f"  渲染 {i + 1}/{args.n_views}")

    np.savez(
        out / "cameras.npz",
        R=np.stack(cam_R).astype(np.float64),
        T=np.stack(cam_T).astype(np.float64),
        centers=np.stack(cam_centers).astype(np.float64),
        fx=np.float64(fx), fy=np.float64(fy), cx=np.float64(cx), cy=np.float64(cy),
        width=np.int64(W), height=np.int64(H),
    )

    rng = np.random.default_rng(args.seed + 999)
    pts, pt_colors = sample_points(arrays, args.n_points, rng, light_dir)
    np.savez(out / "points.npz", points=pts.astype(np.float32), colors=pt_colors.astype(np.float32))

    meta = {
        "n_views": args.n_views, "width": W, "height": H, "fov_x_deg": fov_x,
        "n_points": int(pts.shape[0]), "n_triangles": int(verts.shape[0]),
        "light_dir": light_dir.tolist(), "background": bg.tolist(),
        "note": "GT 由独立的三角形软件渲染器产生，与高斯管线无关",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"完成：{out}（{args.n_views} 视角，{W}x{H}，点云 {pts.shape[0]}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
