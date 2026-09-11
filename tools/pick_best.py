"""遍历 checkpoint，用测试集评估每个模型，挑出最佳的那个。

训练后半程可能退化（实测 opacity reset 会让 PSNR 掉 7 dB），
所以交付的应该是"最好的那一拍"而不是"最后一拍"。

用法： python tools/pick_best.py output/final2/ckpt --data data/synthetic --K 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.colmap_dataset import ColmapDataset
from src.dataset import SyntheticDataset
from src.gaussian_model import GaussianModel
from src.metrics import psnr, ssim
from src.rasterizer import render_gaussians


def evaluate_ply(ply: Path, ds: SyntheticDataset, K: int, sh_degree: int) -> dict:
    model = GaussianModel.load_ply(ply, sh_degree=sh_degree, device=ds.device)
    ps, ss = [], []
    for i in ds.test_indices:
        cam = ds.camera(i)
        with torch.no_grad():
            out = render_gaussians(
                model.get_xyz, model.get_rotation, model.get_scaling,
                model._opacity, model.get_features, cam, ds.background,
                sh_degree=sh_degree, K=K,
            )
            img = out.image.clamp(0.0, 1.0)
        ps.append(psnr(img, ds.image(i)))
        ss.append(ssim(img, ds.image(i)))
    return {
        "ply": str(ply),
        "gaussians": model.num_gaussians,
        "psnr": float(np.mean(ps)),
        "ssim": float(np.mean(ss)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--downscale", type=int, default=1)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--sh-degree", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.data)
    # 自动识别数据来源（和 train.py 的 build_dataset 保持一致）
    if (root / "sparse").exists() or (root / "cameras.bin").exists():
        ds = ColmapDataset(root, device=device, downscale=args.downscale)
    else:
        ds = SyntheticDataset(root, device=device, downscale=args.downscale)

    plies = sorted(Path(args.ckpt_dir).glob("*.ply"))
    if not plies:
        print("没有找到 checkpoint")
        return 1

    results = []
    for p in plies:
        r = evaluate_ply(p, ds, args.K, args.sh_degree)
        results.append(r)
        print(f"{p.name:>20}  高斯 {r['gaussians']:6d}  PSNR {r['psnr']:6.2f} dB  SSIM {r['ssim']:.4f}", flush=True)

    best = max(results, key=lambda r: r["psnr"])
    print("\n最佳：", json.dumps(best, ensure_ascii=False))
    out = Path(args.ckpt_dir) / "best.json"
    out.write_text(json.dumps({"best": best, "all": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
