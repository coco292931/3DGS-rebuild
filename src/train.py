"""3DGS 训练主循环：L1 + D-SSIM、自适应密度控制、球谐升阶。

用法：
    python -m src.train --data data/synthetic --out output/synthetic --iters 3000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .colmap_dataset import ColmapDataset
from .dataset import SyntheticDataset
from .gaussian_model import GaussianModel
from .metrics import d_ssim_loss, psnr, ssim
from .rasterizer import (build_cov3d, compute_colors, near_fade_scale, project_gaussians,
                         rasterize)

# 预览图的近处淡出强度，与 viewer.html 的 FADE_SCALE 默认值保持一致。
# 两边必须用同一个数和同一个基准半径，否则预览图和实际浏览看到的东西对不上。
PREVIEW_FADE_SCALE = 0.22
from .rasterizer_cuda import official_available


def _render_official(*args, **kwargs):
    from .rasterizer_cuda import render_official

    return render_official(*args, **kwargs)

SSIM_LAMBDA = 0.2


def render_view(model: GaussianModel, cam, background, K: int, backend: str = "torch",
                scene_radius: float = 0.0, fade_scale: float = 0.0):
    """渲染一个视角。

    两个后端返回的梯度来源不同，密度控制的阈值不能混用：
      torch 后端 —— 读相机空间坐标 (cam_xy) 的梯度，官方阈值 2e-4 就是配它标定的
      cuda  后端 —— 官方光栅化器只吐屏幕空间 means2D 的梯度，阈值同样是官方 2e-4，
                    但量纲是像素，两者相差 fx/z 倍（本场景约 38 倍）
    """
    if backend == "cuda":
        # +0 让它成为非叶子张量，retain_grad 才能留住梯度——这是官方的写法，
        # 漏掉的话 means2d.grad 恒为 None，密度控制拿不到任何信号（densify 一次都不触发）
        means2d = torch.zeros_like(model.get_xyz, requires_grad=True) + 0
        if means2d.requires_grad:      # evaluate/preview 在 no_grad 下跑，不能 retain
            means2d.retain_grad()
        out, means2d, radii = _render_official(
            model.get_xyz, model.get_rotation, model.get_scaling,
            model._opacity, model.get_features, cam, background,
            sh_degree=model.active_sh_degree, means2d=means2d,
            scene_radius=scene_radius, fade_scale=fade_scale,
        )
        return out, means2d, radii

    colors = compute_colors(model.get_xyz, model.get_features, cam, model.active_sh_degree)
    cov3d = build_cov3d(model.get_rotation, model.get_scaling)
    proj = project_gaussians(model.get_xyz, cov3d, cam)
    if proj.cam_xy.requires_grad:
        proj.cam_xy.retain_grad()
    scale = near_fade_scale(model.get_xyz, cam, scene_radius, fade_scale) if fade_scale > 0 else None
    out = rasterize(proj, colors, model._opacity, cam, background, K=K, opacity_scale=scale)
    return out, proj, proj.radius


@torch.no_grad()
def evaluate(model: GaussianModel, ds, K: int, indices=None, backend: str = "torch"):
    indices = ds.test_indices if indices is None else indices
    psnrs, ssims = [], []
    for i in indices:
        cam = ds.camera(i)
        out, _, _ = render_view(model, cam, ds.background, K, backend)
        img = out.image.clamp(0.0, 1.0)
        gt = ds.image(i)
        psnrs.append(psnr(img, gt))
        ssims.append(ssim(img, gt))
    return float(np.mean(psnrs)), float(np.mean(ssims))


def _write_json_atomic(path: Path, obj) -> None:
    """原子写 JSON。看板会轮询读它，半截文件会直接解析失败。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def save_preview(model, ds, path: Path, K: int, index: int | None = None,
                 train_view: bool = False, backend: str = "torch",
                 scene_radius: float = 0.0, fade_scale: float = 0.0):
    """保存 GT / 渲染 的对比图，便于人眼检查。

    scene_radius + fade_scale 非零时，渲染侧启用相机附近的球形淡出——
    和 WebGL 查看器同一套参数，否则预览图会糊着一层镜头前的高斯，
    和实际浏览时看到的完全不是一个东西。
    """
    if index is None:
        index = ds.train_indices[0] if train_view else ds.test_indices[0]
    with torch.no_grad():
        cam = ds.camera(index)
        out, _, _ = render_view(model, cam, ds.background, K, backend,
                                scene_radius=scene_radius, fade_scale=fade_scale)
        img = out.image.clamp(0.0, 1.0)
    gt = ds.image(index)
    strip = torch.cat([gt, img], dim=2).permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    # 原子写：看板在轮询这个目录，写到一半的 PNG 会被读到
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # PIL 靠扩展名猜格式，.tmp 会直接报 unknown file extension，必须显式给 format
    Image.fromarray((strip * 255.0 + 0.5).astype(np.uint8)).save(tmp, format="PNG")
    os.replace(tmp, path)


def train(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preview").mkdir(parents=True, exist_ok=True)

    if args.backend == "cuda" and not official_available():
        raise RuntimeError("官方 CUDA 光栅化器不可用，请先跑 _upstream/build_dgr.bat")

    ds = build_dataset(args, device)
    if args.limit_views:
        ds.train_indices = ds.train_indices[:args.limit_views]
    print(f"数据：{len(ds.train_indices)} 训练视角 / {len(ds.test_indices)} 测试视角，"
          f"分辨率 {ds.width}x{ds.height}，点云 {ds.points.shape[0]}")

    # 场景尺度取「相机分布的半径」，这是官方 getNerfppNorm 的定义。
    # 不要用点云包围盒：地面会一直延伸到很远，把尺度撑大好几倍，
    # 结果就是 densify 的 split 阈值和 prune 的世界尺度阈值全部失准。
    centers = ds.centers
    radius = float(np.linalg.norm(centers - centers.mean(axis=0), axis=1).max())

    model = GaussianModel(sh_degree=args.sh_degree, device=device)
    model.init_from_points(ds.points, ds.point_colors, scene_extent=radius,
                           scale_factor=args.init_scale_factor)
    model.spatial_lr_scale = radius
    model.training_setup()
    print(f"初始高斯：{model.num_gaussians}，场景尺度（相机半径）{model.scene_extent:.2f}，"
          f"位置学习率 {1.6e-4 * radius:.2e}")

    baseline = ds.mean_image_psnr()
    print(f"瞎猜基线 PSNR（训练集平均图）：{baseline:.2f} dB")

    # 预览用遮罩的基准半径：取点云到中位中心距离的 p85。
    # 刻意不用上面的 radius（那是「相机分布半径」，getNerfppNorm 的定义），
    # 因为 viewer.html 里的 sceneRadius 就是这么算的——两边基准不同的话，
    # 预览图里的淡出范围和实际浏览时看到的会差好几倍。
    _pc = ds.points.detach().float().cpu().numpy().astype(np.float64)   # 可能是 CUDA 张量
    _med = np.median(_pc, axis=0)
    preview_radius = float(np.percentile(np.linalg.norm(_pc - _med, axis=1), 85))
    print(f"预览遮罩基准半径（点云 p85）{preview_radius:.3f}，"
          f"淡出范围 {preview_radius * PREVIEW_FADE_SCALE * 0.15:.3f} ~ "
          f"{preview_radius * PREVIEW_FADE_SCALE:.3f}")

    rng = np.random.default_rng(args.seed)
    log_path = out_dir / "train_log.jsonl"
    log_f = log_path.open("w", encoding="utf-8")
    t_start = time.time()
    ema_loss = None

    for it in range(1, args.iters + 1):
        idx = int(rng.choice(ds.train_indices))
        cam = ds.camera(idx)
        gt = ds.image(idx)

        model.update_learning_rate(it - 1)
        out, proj, radii_from_backend = render_view(model, cam, ds.background, args.K, args.backend)
        img = out.image
        ll1 = torch.abs(img - gt).mean()
        loss = (1.0 - SSIM_LAMBDA) * ll1 + SSIM_LAMBDA * d_ssim_loss(img.unsqueeze(0), gt.unsqueeze(0))

        loss.backward()

        # 屏幕空间位置梯度：判断哪些区域表达不足
        with torch.no_grad():
            if args.backend == "cuda":
                # 官方光栅化器只吐屏幕空间 means2D 的梯度（像素量纲）
                g = proj.grad[:, :2]
                valid = radii_from_backend > 0
                radius = radii_from_backend.float()
            else:
                # 本实现给的是相机空间坐标的梯度
                g = proj.cam_xy.grad
                valid = proj.valid
                radius = proj.radius
            model.add_densification_stats(g, valid)
            model.max_radii2D[valid] = torch.maximum(model.max_radii2D[valid], radius[valid].detach())

        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)

        l = float(loss.detach())
        ema_loss = l if ema_loss is None else 0.98 * ema_loss + 0.02 * l

        # 显存保护：稠密合成张量的规模由片元总数决定，真实场景的高斯会持续膨胀，
        # 实测 1250 步时片元冲到 2300 万（每像素 100 个）直接把 8GB 撑爆。
        # 超限就砍掉屏幕覆盖最大的那批高斯——它们正是贡献片元最多的一批。
        if args.backend == "torch" and out.num_fragments > args.max_fragments:
            n = model.num_gaussians
            k = max(1, int(n * args.emergency_prune_frac))
            _, worst = torch.topk(proj.radius.detach(), k)
            keep = torch.ones(n, dtype=torch.bool, device=device)
            keep[worst] = False
            model._prune_points(keep)
            model._reset_optimizer()
            print(f"[{it:5d}] !! 片元 {out.num_fragments:,} 超过上限，紧急剪掉 {k} 个大高斯 "
                  f"-> {model.num_gaussians}")

        # 自适应密度控制
        if it < args.densify_until and it % args.densify_interval == 0:
            stats = model.densify_and_prune(
                max_grad=args.densify_grad_threshold,
                min_opacity=0.005,
                max_gaussians=args.max_gaussians,
                seed=args.seed + it,
                max_screen_radius=args.max_screen_radius,
                split_radius_px=args.split_radius_frac * max(ds.width, ds.height),
                max_split_frac=args.max_split_frac,
            )
            ps = stats["prune"]
            print(f"[{it:5d}] 增密 {stats['before']} -> clone {stats['clone']} -> split {stats['split']} "
                  f"-> {stats['after']}  |  强制拆 {stats['forced_split']}  |  "
                  f"剪枝 {ps['pruned']}（透明 {ps['low_opacity']} / 屏幕过大 {ps['big_screen']} / "
                  f"世界过大 {ps['big_world']}）")

        # 周期性把不透明度压回去，清掉「又大又透明」的垃圾
        if it < args.densify_until and it % args.opacity_reset_interval == 0:
            model.reset_opacity()
            print(f"[{it:5d}] opacity reset")

        # 球谐升阶：先把低频颜色学好，再逐步放开视角相关项
        if it % args.sh_up_interval == 0 and model.active_sh_degree < model.max_sh_degree:
            model.oneup_sh_degree()
            print(f"[{it:5d}] SH 升到 {model.active_sh_degree} 阶")

        if it % args.log_interval == 0:
            elapsed = time.time() - t_start
            speed = it / elapsed
            rec = {"iter": it, "loss": ema_loss, "gaussians": model.num_gaussians,
                   "fragments": out.num_fragments, "sec_per_iter": round(1.0 / speed, 3)}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            print(f"[{it:5d}] loss={ema_loss:.5f}  高斯={model.num_gaussians:6d}  "
                  f"片元={out.num_fragments:8d}  {speed:.2f} iter/s  已用 {elapsed/60:.1f} 分钟")

        # 定期存 checkpoint：训练后半程可能退化，留着中间模型才能挑最好的交付
        if args.ckpt_interval and it % args.ckpt_interval == 0:
            (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)
            model.save_ply(out_dir / "ckpt" / f"iter_{it:05d}.ply")

        # 实时预览：看板轮询 preview/ 目录拿最新一张，训练中就能看到画面在变好
        pi = args.preview_interval or args.eval_interval
        if pi and it % pi == 0:
            save_preview(model, ds, out_dir / "preview" / f"iter_{it:05d}.png", args.K,
                         train_view=True, backend=args.backend,
                         scene_radius=preview_radius, fade_scale=PREVIEW_FADE_SCALE)

        if it % args.eval_interval == 0 or it == args.iters:
            p, s = evaluate(model, ds, args.K, backend=args.backend)
            tr_p, _ = evaluate(model, ds, args.K, indices=ds.train_indices, backend=args.backend)
            print(f"[{it:5d}] >>> 测试集 PSNR {p:.2f} dB  SSIM {s:.4f}  |  训练集 PSNR {tr_p:.2f} dB"
                  f"  (基线 {baseline:.2f} dB)")

    log_f.close()

    ply_path = out_dir / "point_cloud.ply"
    model.save_ply(ply_path)
    final_psnr, final_ssim = evaluate(model, ds, args.K, backend=args.backend)
    summary = {
        "iters": args.iters,
        "gaussians": model.num_gaussians,
        "test_psnr": final_psnr,
        "test_ssim": final_ssim,
        "baseline_psnr": baseline,
        "resolution": [ds.width, ds.height],
        "minutes": round((time.time() - t_start) / 60.0, 2),
        "ply": str(ply_path),
        "K": args.K,
        "sh_degree": args.sh_degree,
    }
    _write_json_atomic(out_dir / "summary.json", summary)
    print("\n完成：", json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_dataset(args, device):
    """自动判断数据来源：有 COLMAP 模型就用真实数据，否则用合成场景。"""
    root = Path(args.data)
    kind = args.dataset
    if kind == "auto":
        kind = "colmap" if (root / "sparse").exists() or (root / "cameras.bin").exists() else "synthetic"
    if kind == "colmap":
        ds = ColmapDataset(root, device=device, downscale=args.downscale)
        st = ds.filter_stats
        print(f"真实数据：{len(ds.train_indices)} 训练 / {len(ds.test_indices)} 测试视角，"
              f"分辨率 {ds.width}x{ds.height}，相机半径 {ds.scene_extent:.2f}")
        print(f"  点云：原始 {st['raw_points']} -> 过滤后 {st['kept_points']}"
              f"（重投影误差超标 {st['dropped_by_error']}，空间离群 {st['dropped_by_knn']}，"
              f"KNN 阈值 {st['knn_threshold']:.4f}，中心裁剪 {st['prune_radius']:.2f}）")
        print(f"  背景色（图像边框中位数）：{[round(v,3) for v in ds.background.tolist()]}")
        ai = ds.align_info
        print(f"  上方向对齐：旋转 {ai['rotated_deg']:.1f}°，对齐后「图像上方」的 z 分量 "
              f"{ai['up_z_after']:+.3f}（越接近 +1 越好）"
              f"，相机朝向一致性 {ai['camera_consistency']:.3f}")
    else:
        ds = SyntheticDataset(root, device=device, downscale=args.downscale)
    return ds


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="3DGS 训练")
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--dataset", default="auto", choices=["auto", "synthetic", "colmap"])
    ap.add_argument("--backend", default="torch", choices=["torch", "cuda"],
                    help="torch=自写的纯 PyTorch 光栅化器；cuda=官方 diff-gaussian-rasterization")
    ap.add_argument("--out", default="output/synthetic")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--K", type=int, default=12, help="每像素保留的最大高斯数")
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--max-gaussians", type=int, default=60000)
    ap.add_argument("--densify-until", type=int, default=2000)
    ap.add_argument("--densify-interval", type=int, default=200)
    ap.add_argument("--densify-grad-threshold", type=float, default=2e-4)
    ap.add_argument("--split-radius-frac", type=float, default=0.0,
                    help="屏幕半径超过该比例(相对图像长边)的高斯无条件拆开；0 表示关闭")
    ap.add_argument("--max-split-frac", type=float, default=0.25,
                    help="每轮最多拆掉多大比例的高斯，防止一轮拆爆")
    ap.add_argument("--max-fragments", type=int, default=18_000_000,
                    help="片元总数上限，超过就紧急剪枝（防显存爆炸）")
    ap.add_argument("--emergency-prune-frac", type=float, default=0.05)
    ap.add_argument("--max-screen-radius", type=float, default=40.0,
                    help="屏幕半径超过该值的高斯会被剪掉（官方默认 20，那是针对 1600x1200）")
    ap.add_argument("--opacity-reset-interval", type=int, default=1000)
    ap.add_argument("--sh-up-interval", type=int, default=400)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--eval-interval", type=int, default=500)
    ap.add_argument("--ckpt-interval", type=int, default=300)
    ap.add_argument("--preview-interval", type=int, default=0,
                    help="每隔多少步写一张预览图（看板实时预览用；0 表示跟随 eval-interval）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-views", type=int, default=0, help="只用前 N 个训练视角（过拟合诊断）")
    ap.add_argument("--init-scale-factor", type=float, default=1.0,
                    help="初始尺度相对最近邻距离的倍率；越小则每像素片元数越少，K 截断越轻")
    return ap


if __name__ == "__main__":
    train(build_parser().parse_args())
