"""用 pycolmap 跑 SfM：特征提取 -> 匹配 -> 增量式重建。

用法： python tools/run_sfm.py data/real [--stage all|features|match|map]
"""
import argparse
import sys
import time
from pathlib import Path

import pycolmap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--stage", default="all")
    ap.add_argument("--sequential", action="store_true", help="视频序列用顺序匹配，比穷举快得多")
    args = ap.parse_args()

    root = Path(args.root)
    db = root / "database.db"
    img = root / "frames"
    out = root / "sparse"
    out.mkdir(parents=True, exist_ok=True)
    n_frames = len(list(img.glob("*.png")))
    print(f"图像 {n_frames} 张，数据库 {db}")

    if args.stage in ("all", "features"):
        t = time.time()
        # 视频抽帧没有 EXIF，默认 AUTO 模式会把每一帧当成一台独立相机，
        # 导致内参各自自由拟合、位姿不稳。这里强制所有帧共用一个相机。
        try:
            mode = pycolmap.CameraMode.SINGLE
        except AttributeError:
            mode = "SINGLE"
        pycolmap.extract_features(str(db), str(img), camera_mode=mode)
        print(f"特征提取完成 {time.time()-t:.1f}s", flush=True)

    if args.stage in ("all", "match"):
        t = time.time()
        if args.sequential:
            pycolmap.match_sequential(str(db))
        else:
            pycolmap.match_exhaustive(str(db))
        print(f"匹配完成 {time.time()-t:.1f}s", flush=True)

    if args.stage in ("all", "map"):
        t = time.time()
        maps = pycolmap.incremental_mapping(str(db), str(img), str(out))
        print(f"重建完成 {time.time()-t:.1f}s，得到 {len(maps)} 个模型", flush=True)
        if not maps:
            print("!!! SfM 失败：没有重建出任何模型")
            return 1
        best = max(maps.values(), key=lambda m: m.num_reg_images())
        print(f"最佳模型：{best.num_reg_images()} 张图注册成功 / {best.num_points3D()} 个三维点")
        print(f"  平均重投影误差 {best.compute_mean_reprojection_error():.3f} px")
        if best.num_reg_images() < n_frames * 0.5:
            print(f"  !! 只有 {best.num_reg_images()}/{n_frames} 帧注册成功，位姿质量堪忧")
        # 模型本身已经写在 sparse/ 下（ColmapDataset 会自动挑注册图像最多的那个）。
        # 这里只是额外导出一份便于外部工具读取；失败不该中断流程——
        # 之前这里没建目录，直接抛 ValueError 把整条导入链路带崩过。
        try:
            extra = root / "best_model"
            extra.mkdir(parents=True, exist_ok=True)
            best.write(str(extra))
            print(f"已写出 {extra}")
        except Exception as e:
            print(f"（附加导出失败，不影响后续训练：{str(e)[:120]}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
