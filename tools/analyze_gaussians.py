"""高斯数变化 与 loss 平台期的关系。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for run in ["output/_probe_curve", "output/cuda_long", "output/cuda_run"]:
    p = ROOT / run / "train_log.jsonl"
    if not p.exists():
        continue
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    it = [r["iter"] for r in rows]
    loss = [r["loss"] for r in rows]
    g = [r.get("gaussians", 0) for r in rows]
    print(f"=== {run}  ({len(rows)} 点, 间隔 {it[1]-it[0] if len(it)>1 else '?'} 步) ===")
    print(f"{'步数':>7} {'loss':>9} {'高斯数':>8}")
    step = max(1, len(it) // 12)
    for i in range(0, len(it), step):
        print(f"{it[i]:7d} {loss[i]:9.5f} {g[i]:8d}")

    # 高斯数首次连续 5 点不变的步数
    first_stable = None
    for i in range(4, len(g)):
        if all(g[j] == g[i] for j in range(i - 4, i + 1)):
            first_stable = it[i]
            break
    print(f"  -> 高斯数首次稳定于第 {first_stable} 步")

    # 稳定前后的 loss 平均下降速度
    if first_stable:
        k = next((i for i in range(len(it)) if it[i] >= first_stable), None)
        if k and k + 1 < len(it):
            before = (loss[0] - loss[k]) / max(k, 1) * 1000
            after = (loss[k] - loss[-1]) / max(len(it) - k, 1) * 1000
            print(f"  -> 稳定前每千步降 {before:.5f}（高斯在变）")
            print(f"  -> 稳定后每千步降 {after:.5f}")
    print()
