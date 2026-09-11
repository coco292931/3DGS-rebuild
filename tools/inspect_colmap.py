"""查看 COLMAP 重建结果：相机内参、位姿、点云规模。"""
import sys
from pathlib import Path

import numpy as np
import pycolmap

root = Path(sys.argv[1])
rec = pycolmap.Reconstruction(str(root))
print(f"图像 {rec.num_reg_images()} / 相机 {rec.num_cameras()} / 三维点 {rec.num_points3D()}")

for cam_id, cam in rec.cameras.items():
    print(f"\n相机 #{cam_id}: model={cam.model_name}  {cam.width}x{cam.height}")
    print(f"  focal  {cam.focal_length_x:.2f}, {cam.focal_length_y:.2f}")
    print(f"  center {cam.principal_point_x:.2f}, {cam.principal_point_y:.2f}")
    print(f"  全部参数 {np.round(np.array(cam.params), 5)}")
    # SIMPLE_RADIAL 的参数顺序是 (f, cx, cy, k)，畸变 k 在最后一位
    if len(cam.params) >= 4:
        print(f"  >>> 径向畸变 k1 = {cam.params[3]:+.5f}")

print("\n前 3 张图的位姿：")
imgs = sorted(rec.images.items())[:3]
for img_id, img in imgs:
    pose = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
    t = np.array(pose.translation)
    R = np.array(pose.rotation.matrix())
    c = -R.T @ t
    print(f"  {img.name}: 相机中心 {np.round(c, 3)}  |t| {np.linalg.norm(t):.3f}")

# 点云范围
pts = np.array([p.xyz for p in rec.points3D.values()])
print(f"\n点云范围 min {np.round(pts.min(0), 3)}  max {np.round(pts.max(0), 3)}")
print(f"点云对角 {np.linalg.norm(pts.max(0) - pts.min(0)):.3f}")
