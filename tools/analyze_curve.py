"""分析 loss 曲线为什么波动、checkpoint 效果为什么不线性。

把 loss 记录和训练事件（opacity reset / 增密 / SH 升阶）对齐，看相关性。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
run = sys.argv[1] if len(sys.argv) > 1 else "output/_probe_curve"
log = ROOT / run / "train_log.jsonl"
rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"{run}：{len(rows)} 个记录点，步数 {rows[0]['iter']} ~ {rows[-1]['iter']}")

it = np.array([r["iter"] for r in rows], dtype=float)
loss = np.array([r["loss"] for r in rows], dtype=float)
g = np.array([r.get("gaussians", 0) for r in rows], dtype=float)

# 1) 局部波动：相邻点的相对变化
d = np.abs(np.diff(loss)) / np.maximum(loss[:-1], 1e-9)
print(f"\n[1] 相邻记录点 loss 相对变化：中位 {np.median(d)*100:.2f}%  最大 {d.max()*100:.2f}%")
big = np.where(d > 0.05)[0]
print(f"    跳变 >5% 的点数：{len(big)} / {len(d)}")
for i in big[:12]:
    print(f"      step {int(it[i]):6d} -> {int(it[i+1]):6d}   loss {loss[i]:.5f} -> {loss[i+1]:.5f}"
          f"  ({(loss[i+1]-loss[i])/loss[i]*100:+.1f}%)")

# 2) 平滑趋势 vs 实际：算 11 点滑动平均，看偏差
k = 11
if len(loss) > k:
    sm = np.convolve(loss, np.ones(k) / k, mode="same")
    resid = loss - sm
    print(f"\n[2] 相对平滑趋势的偏差：标准差 {resid.std():.5f}，"
          f"最大 {np.abs(resid).max():.5f}（loss 量级 {loss.mean():.5f}）")

# 3) 高斯数变化
dg = np.diff(g)
n_change = int((dg != 0).sum())
print(f"\n[3] 高斯数：{int(g[0])} -> {int(g[-1])}，发生过 {n_change} 次变化"
      f"（占总记录点 {n_change/len(dg)*100:.1f}%）")

# 4) 分段看 loss 下降速度
print("\n[4] 分段下降速度（每 1000 步的 loss 降幅）：")
for a in range(0, int(it[-1]), 1000):
    m = (it >= a) & (it < a + 1000)
    if m.sum() < 2:
        continue
    seg = loss[m]
    drop = (seg[0] - seg[-1]) / max(seg[0], 1e-9) * 100
    print(f"    {a:5d}-{a+1000:5d}: loss {seg[0]:.5f} -> {seg[-1]:.5f}  降 {drop:+6.2f}%"
          f"   高斯 {int(g[m][0])}->{int(g[m][-1])}")
