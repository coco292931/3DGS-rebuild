"""造一个颜色/位置都已知的测试模型，用来判定查看器的坐标朝向是否正确。

约定（与训练端一致）：世界 z 朝上、x 朝右、y 朝里。
    +z 红   -z 蓝   +x 绿   -x 品红   +y 黄   -y 青
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gaussian_model import GaussianModel

R = 1.0
probes = [
    ([0, 0, R], [1.0, 0.0, 0.0]),      # +z 上看   -> 红
    ([0, 0, -R], [0.0, 0.0, 1.0]),     # -z 下看   -> 蓝
    ([R, 0, 0], [0.0, 1.0, 0.0]),      # +x 右看   -> 绿
    ([-R, 0, 0], [1.0, 0.0, 1.0]),     # -x 左看   -> 品红
    ([0, R, 0], [1.0, 1.0, 0.0]),      # +y 远看   -> 黄
    ([0, -R, 0], [0.0, 1.0, 1.0]),     # -y 近看   -> 青
]

# 每个轴上放一串小球，方便看出朝向
pts, cols = [], []
for (c, col) in probes:
    for t in (0.35, 0.6, 0.85, 1.0):
        pts.append([c[0] * t, c[1] * t, c[2] * t])
        cols.append(col)

model = GaussianModel(sh_degree=0, device="cpu")
model.init_from_points(torch.tensor(pts, dtype=torch.float32),
                       torch.tensor(cols, dtype=torch.float32))
with torch.no_grad():
    model._scaling.fill_(torch.log(torch.tensor(0.06)).item())
    model._opacity.fill_(4.0)          # sigmoid(4)≈0.982，确保清晰可见
out = Path(sys.argv[1])
model.save_ply(out)
print(f"写出 {out}：{model.num_gaussians} 个探针高斯")
print("  红=+z上  蓝=-z下  绿=+x右  品红=-x左  黄=+y远  青=-y近")
