"""检测 SfM 世界坐标系的朝向：相机 y 轴（图像向下）在世界里指向哪边。

如果所有相机的「图像下方」平均指向世界 +z，说明世界 +z 其实是"下"，场景整个是倒的。
"""
import sys
from pathlib import Path

import numpy as np
import pycolmap

rec = pycolmap.Reconstruction(sys.argv[1])
ys, zs, centers = [], [], []
for img in rec.images.values():
    pose = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
    R = np.asarray(pose.rotation.matrix())
    t = np.asarray(pose.translation)
    ys.append(R[1])                    # 相机 y 轴 = 图像向下
    zs.append(R[2])                    # 相机 z 轴 = 视线前方
    centers.append(-R.T @ t)
ys = np.array(ys); zs = np.array(zs); centers = np.array(centers)
mean_y = ys.mean(axis=0)
mean_z = zs.mean(axis=0)
print(f"相机数 {len(ys)}")
print(f"图像下方在世界中的平均方向 : {np.round(mean_y, 4)}")
print(f"视线前方在世界中的平均方向 : {np.round(mean_z, 4)}")
print(f"相机中心均值               : {np.round(centers.mean(axis=0), 3)}")
print()
print(f">>> 图像下方 · 世界z = {mean_y[2]:+.4f}")
if mean_y[2] > 0.15:
    print(">>> 判定：世界 +z 指向画面下方 —— 场景上下颠倒，需要翻转")
elif mean_y[2] < -0.15:
    print(">>> 判定：世界 +z 指向画面上方 —— 朝向正确")
else:
    print(">>> 判定：相机朝向分散，单靠 z 分量不足以判断")

# 点云的主轴：木桩+桌面应当有个明显的「薄」方向
pts = np.array([p.xyz for p in rec.points3D.values()])
c = pts.mean(axis=0)
cov = np.cov((pts - c).T)
w, v = np.linalg.eigh(cov)
print(f"\n点云主轴特征值 {np.round(w, 4)}")
print(f"最小主轴方向   {np.round(v[:, 0], 4)}（平面法线候选）")
print(f"该轴 · 世界z   = {v[0, 2]:+.4f}")
