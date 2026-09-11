"""把 loss 的上升点和训练事件（opacity reset / 增密 / SH 升阶）对齐。"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
run = sys.argv[1] if len(sys.argv) > 1 else "output/_probe_curve"
rows = [json.loads(l) for l in (ROOT / run / "train_log.jsonl").read_text(
    encoding="utf-8").splitlines() if l.strip()]
it = np.array([r["iter"] for r in rows], dtype=float)
loss = np.array([r["loss"] for r in rows], dtype=float)
g = np.array([r.get("gaussians", 0) for r in rows], dtype=float)

print("=== 所有 loss 上升（变差）的点 ===")
up = np.where(np.diff(loss) > 0)[0]
for i in up:
    print(f"  step {int(it[i]):6d} -> {int(it[i+1]):6d}   "
          f"{loss[i]:.5f} -> {loss[i+1]:.5f}  ({(loss[i+1]-loss[i])/loss[i]*100:+.2f}%)")

# 事件表
print("\n=== 训练事件 ===")
print("  opacity reset  : 每 1500 步 -> ", [1500, 3000, 4500])
print("  SH 升阶        : 每 1000 步（0->1->2->3 阶）")
print("  增密           : 前 3000 步，每 100 步一次")
print("  高斯数变化点   :", [int(it[i+1]) for i in np.where(np.diff(g) != 0)[0]][:20])

# 关键对比：reset 步附近 vs 非 reset 步附近的平均变化
print("\n=== reset 步附近 vs 其他位置的 loss 变化 ===")
def stat(near):
    idx = [i for i in range(len(loss) - 1)
           if near(int(it[i]), int(it[i+1]))]
    if not idx:
        return None
    v = (loss[[i + 1 for i in idx]] - loss[idx]) / loss[idx] * 100
    return len(v), v.mean(), v.std(), v.min(), v.max()

r1 = stat(lambda a, b: any(x in range(a, b + 1) for x in (1500, 3000, 4500)))
r2 = stat(lambda a, b: not any(x in range(a, b + 1) for x in (1500, 3000, 4500)))
for name, r in (("跨越 reset 的区间", r1), ("其他区间", r2)):
    if r:
        print(f"  {name}: n={r[0]:3d}  平均 {r[1]:+.2f}%  标准差 {r[2]:.2f}%  "
              f"范围 [{r[3]:+.2f}%, {r[4]:+.2f}%]")
