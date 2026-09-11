"""分析 SfM 点云：重投影误差、轨迹长度、空间离群程度，确定过滤阈值。"""
import sys
from pathlib import Path

import numpy as np
import pycolmap

rec = pycolmap.Reconstruction(sys.argv[1])
pts = []
for p in rec.points3D.values():
    pts.append([*p.xyz, p.error, p.track.length()])
pts = np.array(pts)
xyz, err, track = pts[:, :3], pts[:, 3], pts[:, 4]
print(f"点数 {len(pts)}")
print(f"重投影误差 err: p50={np.percentile(err,50):.3f} p90={np.percentile(err,90):.3f} "
      f"p99={np.percentile(err,99):.3f} max={err.max():.2f}")
print(f"轨迹长度 track: p10={np.percentile(track,10):.0f} p50={np.percentile(track,50):.0f} "
      f"p90={np.percentile(track,90):.0f} max={track.max():.0f}")

# 空间离群：到最近邻的距离（采样后算，避免 O(N^2)）
from scipy.spatial import cKDTree
idx = np.random.default_rng(0).choice(len(xyz), min(4000, len(xyz)), replace=False)
tree = cKDTree(xyz)
d, _ = tree.query(xyz[idx], k=6)
knn = d[:, 1:].mean(axis=1)
print(f"\nKNN 平均距离: p50={np.percentile(knn,50):.4f} p90={np.percentile(knn,90):.4f} "
      f"p99={np.percentile(knn,99):.4f} max={knn.max():.3f}")

# 到点云中位中心的距离
med = np.median(xyz, axis=0)
rad = np.linalg.norm(xyz - med, axis=1)
print(f"到中位中心距离: p50={np.percentile(rad,50):.3f} p90={np.percentile(rad,90):.3f} "
      f"p99={np.percentile(rad,99):.3f} max={rad.max():.2f}")

print("\n=== 各种过滤组合下保留的点数 ===")
for e_thr in (0.5, 1.0, 1.5, 2.0):
    for t_thr in (3, 5, 8):
        m = (err < e_thr) & (track >= t_thr)
        m2 = m & (rad < np.percentile(rad, 99.5))
        print(f"  err<{e_thr} & track>={t_thr}: {m.sum():6d} -> 再去掉最远 0.5%: {m2.sum():6d}")
