"""统计 PLY 里高斯的关键量分布，判断「画不出来」是不是因为不透明度太低。"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gaussian_model import read_ply_vertex_data


def main(path):
    names, data = read_ply_vertex_data(Path(path))
    idx = {n: i for i, n in enumerate(names)}
    op = data[:, idx["opacity"]]
    opacity = 1.0 / (1.0 + np.exp(-op))
    s = np.stack([np.exp(data[:, idx[f"scale_{i}"]]) for i in range(3)], axis=1)
    print(f"高斯数 {data.shape[0]}")
    for q in (1, 5, 25, 50, 75, 95, 99):
        print(f"  opacity p{q:02d} = {np.percentile(opacity, q):.4f}")
    print(f"  opacity >= 0.002 的比例: {(opacity >= 0.002).mean():.3f}")
    print(f"  opacity >= 0.05  的比例: {(opacity >= 0.05).mean():.3f}")
    print(f"  opacity >= 0.3   的比例: {(opacity >= 0.3).mean():.3f}")
    print(f"  scale 中位数 {np.median(s):.4f}  最大 {s.max():.4f}")
    print(f"  位置范围 {data[:, idx['x']].min():.2f}..{data[:, idx['x']].max():.2f} "
          f"{data[:, idx['y']].min():.2f}..{data[:, idx['y']].max():.2f} "
          f"{data[:, idx['z']].min():.2f}..{data[:, idx['z']].max():.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
