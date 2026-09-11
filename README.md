# 3D Gaussian Splatting 复刻实现

> 从零实现的 3D Gaussian Splatting（Kerbl et al., SIGGRAPH 2023），**纯 PyTorch 可微光栅化器**，
> 不依赖官方 CUDA 扩展。合成场景与真实拍摄视频两条链路均已跑通，配套单文件 WebGL 查看器，
> 可自由移动视角浏览。

## 结果一览

| 场景 | 数据来源 | 测试集 PSNR | SSIM | 高斯数 | 训练成本 |
|---|---|---|---|---|---|
| 合成场景 | 程序化生成（4 物体 + 棋盘地面） | **23.77 dB** | **0.929** | 24,075 | 2000 步 / 12 分钟 / 640×480 |
| 真实场景 | 实拍木桩视频 + SfM | **22.51 dB** | **0.849** | 31,830 | 3000 步 / 11 分钟 / 640×360 |

两条链路的「瞎猜基线」（拿训练集平均图当预测）分别是 15.71 dB 和 15.94 dB。

---

## 一、和官方实现的差异

| | 官方 (graphdeco-inria) | 本实现 |
|---|---|---|
| 光栅化 | 手写 CUDA 核（diff-gaussian-rasterization） | 纯 PyTorch，向量化分桶 + autograd 反向 |
| 并行策略 | 16×16 tile 分块，块内排序，**无 K 上限** | 按像素分桶，每像素保留最近 **K** 个 |
| 反向传播 | 手写反向核 | autograd 自动求导 |
| 依赖 | CUDA 工具链 + 编译 | 只需要 torch |
| 速度 | 10~30 ms/迭代 | 31 ms/迭代（640×480, K=64） |

### 主动接受的取舍

**分桶 vs 分块**：分桶没有 Python 循环、完全向量化、显存可控，代价是高斯极端交叠处会丢弃
第 K 个之后的贡献。这是本实现**一切质量问题的总根源**——详见「七、已知限制」。

**反向靠 autograd**：慢，但改一行公式不需要同步改反向核。

光栅化器按可替换后端设计（`project_gaussians` → `rasterize` 两段式接口），
接 CUDA 后端时上层训练代码无需改动。

---

## 二、快速开始

### 2.1 环境

| 项 | 实测通过的版本 |
|---|---|
| GPU | RTX 5060 8.5GB（sm_120） |
| PyTorch | 2.9.1+cu130 |
| Python | 3.10 |
| 其他 | numpy / scipy / pillow / pycolmap（真实场景需要） |

```bash
pip install numpy scipy pillow
pip install pycolmap      # 仅真实场景需要，PyPI 上有 win_amd64 轮子，无需编译 COLMAP
```

> 本实现本身**不需要** CUDA 工具链。若要编译官方的 CUDA 光栅化器，
> 配置见「六、踩坑记录」中的工具链一节。

### 2.2 跑通合成场景

```bash
# 1) 生成合成场景（GT 由独立的三角形软件渲染器产生）
python tools/synth_scene.py --out data/synthetic --n-views 80 --width 640 --height 480

# 2) 训练
python -m src.train --data data/synthetic --iters 2000 --downscale 1 --K 64 \
    --out output/mine --init-scale-factor 0.5

# 3) 看结果：双击 output/deliverable/view.bat，浏览器里拖拽旋转 / 滚轮缩放 / WASD 平移
```

### 2.3 跑通真实场景

```bash
# 1) 抽帧。fps 必须够密：手持转动 0.5 秒能转过二十几度，帧间视差过大会直接匹配失败。
#    实测同一段视频：fps=2 抽 70 帧只注册 26 张；fps=6 抽 210 帧注册 202 张（96%）
ffmpeg -i video.mp4 -vf "fps=6,scale=1280:-1" -q:v 2 data/real/frames/frame_%04d.png

# 2) SfM（pycolmap）。视频抽帧没有 EXIF，必须强制单相机模式，
#    否则每帧会被当成独立相机各自拟合内参，位姿不稳。
python tools/run_sfm.py data/real --stage features
python tools/run_sfm.py data/real --stage match --sequential
python tools/run_sfm.py data/real --stage map

# 3) 训练（自动识别 COLMAP 数据，无需手动转换格式）
python -m src.train --data data/real --dataset colmap --downscale 2 --K 64 \
    --iters 3000 --out output/mine-real --init-scale-factor 0.3
```

---

## 三、验证方式（凭什么说这个实现是对的）

### 3.1 GT 与训练管线完全独立

合成场景的参考图由**三角形软件光栅化器**（z-buffer + 透视正确插值 + Lambert 光照）产生，
与高斯管线的数学没有交集。刻意不走「用高斯渲染 GT 再用高斯去拟合」那条路——那样 PSNR 再高
也证明不了任何事。

### 3.2 光栅化器有独立基准

`rasterizer.py` 里同时提供了**朴素参考实现**（逐高斯循环、全图 alpha 累加、逻辑直白到不可能写错）。
向量化实现必须复现它，且两者共用同一套包围盒裁剪，保证比较的是分桶/排序/合成逻辑本身。

### 3.3 梯度用有限差分验证，不靠 autograd 自证

双精度下 eps=1e-5 时，位置/缩放/不透明度/颜色的解析梯度与数值梯度相对误差在 1e-8 ~ 1e-4 量级。

**注意**：有限差分必须在 float64 下做。float32 时 loss 的变化量已淹没在浮点精度里，
实测同样 eps 下相对误差会退化到 0.5 量级——那是差分方法失效，不是实现错误。

### 3.4 查看器用真实鼠标事件验证

用 selenium 走浏览器真实输入管线（mousedown → mousemove → mouseup），拖拽 150px 后
`cam.yaw` 恰好变化 0.900，与代码里 `yaw -= dx*0.006` 逐位对得上。

跑法：

```bash
python -m pytest tests -q      # 12 项全部通过
```

---

## 四、真实数据特有的处理

### 4.1 点云三重过滤

SfM 输出的点云不能直接用。实测 30741 个点里：

| 过滤条件 | 剔除数 |
|---|---|
| 重投影误差 > 1.5px | 807 |
| 空间离群（5 近邻平均距离 > p97） | 1549 |
| 距中心 > 3× 相机半径 | 剩余兜底 |

**空间离群最致命**：KNN 距离的 p99 是 p50 的 **19 倍**，这些稀疏点变成高斯后会甩到远处，
把屏幕覆盖面积撑爆。只看「到中心距离」是不够的。

### 4.2 上方向对齐（必做，不是可选项）

**SfM 的世界坐标系由前两帧决定，和「z 朝上」没有任何约定关系。** 实测两段不同视频：

| 视频 | 图像下方 · 世界z | 结论 |
|---|---|---|
| VID_20260910_214703 | +0.4624 | 倒 |
| VID_20260910_230203 | +0.3409 | **也倒** |

两段都是倒的，所以在 `ColmapDataset` 里自动对齐：用「相机图像上方的平均方向」定 up
（人不会倒着拿手机，这个信号比点云主平面法线可靠——后者会被离群点带偏），
再构造旋转把 up 转到 +z。

两个必须做对的地方（都踩过）：

1. **矩阵乘顺序**：新世界坐标 `p' = R_align p`，代回 `p_cam = R p + t` 得
   `p_cam = (R @ R_align^T) p' + t`，所以是 **`R_new = R_old @ R_align^T`**。
   写成 `R_align @ R_old` 是把旋转作用在相机轴上，场景根本不会转正。
2. **点云要跟着转**：只转相机会让初始高斯留在旧坐标系，和相机对不上，
   投影到屏幕的片元数会从 **400 万掉到 11 万**，训练直接废掉。

### 4.3 内参与等效焦距的核对

用「26mm 等效焦距」（镜头标称）反推像素焦距 882.5px（@1280 宽），SfM 估出 996.8px，
**SfM 长 12.9%**。这不是 SfM 错了：手机录像会为电子防抖裁切画面（典型 10~20%），
裁切后视野变窄、等效焦距变长，视频的实际等效焦距约 29.4mm。
若强行按 26mm 固定内参，反而会把重建搞歪。

---

## 五、性能：一次 22 倍的优化

**现象**：backward 占每步耗时的 99%（686ms / 705ms），但逐操作 micro-benchmark 测出整个合成链
只要 14ms。差 35 倍。

**根因**：稠密 `[像素数 × K]` 桶里约一半槽位是空的，而 `(sparse - 1).clamp(min=0)` 把它们
**全部指向索引 0**。这些槽位的 alpha 恒为 0、梯度恒为 0，指向哪里都不影响正确性，
**但反向的 scatter-add 仍会为它们执行原子加**——第 0 个高斯因此承受约 75 万次原子加，
把并行 scatter 串行化了。

| 索引分布 | backward 耗时 |
|---|---|
| 均匀随机 | 5.5 ms |
| **全指向索引 0（修复前）** | **634 ms** |
| 分散索引（修复后） | 5.6 ms |

修法是 `torch.where(valid_slot, gidx, arange(P*K) % N)`。

| | 修复前 | 修复后 |
|---|---|---|
| backward | 686 ms | **20.7 ms** |
| 训练速度 | 1.4 iter/s | **31.6 iter/s** |

**这个优化不是锦上添花，而是质量的前提**：省下的时间才开得起 K=64，而 K 不够时训练会直接发散
（见下一节）。

> 排查中被推翻的三个猜测：索引冲突（真实 vs 均匀 0.92/0.90ms，一样）、batched 3×3 矩阵乘（1.1ms）、
> 显存压力（峰值仅 388MB）。**逐操作 micro-benchmark 是测不出这类问题的**——差异全部来自
> 索引分布这种数据相关的退化，只有跑真实数据、真实代码才测得出来。

---

## 六、踩坑记录

### 训练逻辑

- **K 截断会让训练发散，不是画质略降**。被困住的高斯拿不到梯度，优化器转而把没被截断的
  高斯撑得更大去补窟窿，片元数进一步增长，形成正反馈。实测（640×480，其余参数相同）：

  | K | 500 步 | 1000~1500 步 | 结果 |
  |---|---|---|---|
  | 20 | 21.95 dB | 18.40 dB | 崩 |
  | 32 | 22.37 dB | 19.63 dB | 崩 |
  | 64 | 23.09 dB | **23.49 dB** | 稳定上升 |

- **K 截断的解药是提高分辨率**，不是收紧剪枝。训练中高斯的屏幕覆盖会持续增长
  （实测每个高斯从覆盖 230 像素涨到 349，片元总数 1.7M→8.4M），因为优化器倾向用更大的高斯
  换更低的 loss。每像素片元数 = 片元总数 / 像素数，提高分辨率直接摊薄它：
  320×240 时每像素 109 个片元（K=32 不够），640×480 时降到 29 个（K=32 富余）。
  曾试过用 `max_screen_radius` 压回去，结果把近处地面点剪光、地面整个消失——那是错误的方向。
- **`max_screen_size` 官方默认是不剪的**，别自作多情地设小值。官方 `prune` 的屏幕半径上限
  默认 `None`，只有手动传入才启用。近处地面点的屏幕投影天然就大，设成 12~25 会把它们成片剪掉。
- **`opacity reset` 在小规模训练里会把训练打崩**。800 步触发时测试集 PSNR 从 19.7 直接掉到 12.3，
  且后续 700 步都没恢复。官方能用是因为后面还有两万步。
- **场景尺度必须取「相机分布半径」**（官方 `getNerfppNorm` 的定义）。用点云包围盒会被地面
  撑大好几倍，densify / prune 的所有阈值一起失准。
- **密度控制的梯度必须用相机空间坐标**。官方阈值 2e-4 是配合相机空间梯度标定的；换成屏幕坐标
  梯度会差 `fx/z` 倍（本场景约 38 倍），阈值永远够不到，densify 一次都不会触发。
- **初始不透明度是 `inverse_sigmoid(0.1) ≈ -2.197`，不是 0.1**。直接把 0.1 当 logit 用，
  实际初始不透明度变成 `sigmoid(0.1)=0.525`，画面一片糊。
- **`torch.cat` / 索引 / 任何算子的输出都不是叶子张量**，塞进优化器会报
  `can't optimize a non-leaf Tensor`。每次重建参数张量都必须 detach。
- 训练期统计量（`max_radii2D` 等）必须跟参数一起增删，否则下一次剪枝掩码长度对不上。

### 查看器

- **`alpha: false` 的 WebGL context 没有 alpha 通道，`DST_ALPHA` 恒为 1**，
  于是混合因子 `ONE_MINUS_DST_ALPHA` 恒为 0，所有片元贡献被乘成零：画面一片空白，
  而 `gl.getError()` 依然是 0，GPU 还在正常光栅化（帧率骤降但无输出）。
  从远到近排序的合成应该用 `blendFunc(ONE, ONE_MINUS_SRC_ALPHA)` + 预乘颜色。
- **相机空间的 z 方向约定必须和投影矩阵一致**。本实现的相机是 +z 朝前（与训练端一致），
  照抄 OpenGL 的透视矩阵（假设 -z 朝前、y 朝上）会让 NDC.z 落到 [-1,1] 之外，整个四边形被裁掉。
  深度映射取 `A + B/z` 时 A 必须为正。
- **相机贴近时的糊屏**用球形渐变解决：距离淡出（半径 `sceneRadius × 0.22`）+ 屏幕尺寸淡出
  （标准差超过视口 8% 开始淡、30% 全透明，60% 硬剔除兜底）。都是渐变而非硬裁剪——
  硬裁剪会让高斯越过阈值那一帧突然闪一下。
  `?fade=0` 可关闭。**注意**：关闭时不能把半径设成 0，`smoothstep(0,0,z)` 行为不确定，
  实测让所有高斯 alpha 变成 NaN 被丢弃、画面全空；传一对负边界才对。
- 单目 SfM 没有绝对尺度，**自动摆位不能假设场景在原点**。用点云到中位中心距离的 p85 分位数，
  用 AABB 会被离群点撑大几倍、相机飞到十万八千里外。
- 验证这类问题不能靠读代码：最终定位靠的是「在 `draw()` 之后的同一帧内 `readPixels`」——
  截到清屏色就说明一个像素都没写上，直接把范围从渲染逻辑缩到混合/裁剪。

### CUDA 工具链（若要编译官方扩展）

1. `pip install ninja` —— torch 没有 ninja 直接拒绝编译。
2. **CUDA 12.8 + torch cu130 编不过**：`compiled_autograd.h: error C2872: std 不明确的符号`。
   换 MSVC 14.29 同样失败。
3. **CUDA 13.4 + MSVC 14.44 仍失败**：CCCL 强制要求标准预处理器，
   报 `fatal error C1189: MSVC/cl.exe with traditional preprocessor is used`。
   加上 `-Xcompiler /Zc:preprocessor` 后编译通过、核函数在 sm_120 上正确执行。
4. DSH 沙箱在 workspace-write 模式下禁止子进程建管道，`subprocess.run(capture_output=True)`
   报 `WinError 5`，而 torch 检测 ninja、调用 nvcc 全走管道，必须用 full-access。

验证脚本见 `_env_probe/cuda_toolchain_probe.py`（`verdict: TOOLCHAIN_WORKS`）。

---

## 七、已知限制

1. **训练在 1500 步左右见顶后回落**。根因是 densification 让高斯持续膨胀，片元数不断增长，
   最终撞上 K 上限，随后触发紧急剪枝、PSNR 掉 2~3 dB 回不去。
   **根治方向**：tile 分块重写光栅化器（消除 K 上限），或接 CUDA 后端。
2. **速度**：纯 PyTorch 比官方 CUDA 实现慢约一个量级。
3. **规模**：稠密合成张量限制了 `分辨率 × 高斯数` 的乘积，8.5GB 卡上实用上限约 640×480 / 3 万高斯。
4. **优化器动量**：每次密度控制后重建 Adam，动量会丢失（官方会搬运 optimizer state）。
5. **查看器只用球谐 DC 分量**做颜色，不含视角相关的高阶项。
6. **真实场景的漂浮高斯**：桌面之外的背景杂物重建得很碎，这是视角覆盖不均 + 训练步数不足的双重结果。
7. **评估口径偏窄**：只看平均 PSNR 会掩盖视角方差（实景三个测试视角分别是 22.80 / 28.02 / **12.04** dB，
   平均值完全不能反映「其中三分之一是废的」）。应当补上 PSNR 的**最差 10% 分位**与感知指标。

---

## 八、目录结构

```
src/
  sh.py                 三阶实球谐求值
  camera.py             针孔相机模型
  rasterizer.py         核心：投影 + 分桶 + 深度排序 + alpha 合成（含朴素参考实现）
  gaussian_model.py     高斯参数、激活、初始化、PLY 读写、自适应密度控制
  dataset.py            合成场景数据集
  colmap_dataset.py     真实场景数据集（含点云过滤与上方向对齐）
  metrics.py            PSNR / SSIM
  train.py              训练主循环
tools/
  synth_scene.py        合成场景生成（独立的三角形软件渲染器）
  run_sfm.py            pycolmap 的 SfM 封装
  pick_best.py          用测试集评估所有 checkpoint，挑最佳（训练后半程会退化，不能交付最后一拍）
  shot_drag.py          真实鼠标事件驱动查看器截图（验证交互链路）
  shot_viewer.py        单张截图
  shot_closeup.py       逐档推近相机，验证近处淡出
  explore_angles.py     批量扫视角
  real_compare.py       真实场景 GT vs 渲染对比
  analyze_points.py     点云分布分析（定过滤阈值）
  check_orientation.py  检测 SfM 世界坐标系朝向
  inspect_colmap.py     查看 COLMAP 模型内容
  ply_stats.py          PLY 统计
  bench_step.py         分段计时
  test_zeroconflict.py  性能问题的最小复现（空槽位索引冲突）
viewer/
  viewer.html           单文件 WebGL2 查看器，零依赖
tests/
  test_rasterizer.py    光栅化器验证（8 项）
  test_model.py         参数模型验证（4 项）
output/
  deliverable/          合成场景成品（PLY + 查看器 + 截图 + view.bat）
  deliverable_real/     真实场景成品
_env_probe/             环境探测脚本与结果记录
```

---

## 九、交互说明

打开 `output/deliverable/view.bat`（或 `deliverable_real/view.bat`），浏览器自动打开。

| 操作 | 效果 |
|---|---|
| 左键拖拽 | 旋转视角 |
| 滚轮 | 缩放 |
| WASD | 平移 |
| Q / E | 升降 |
| Shift | 加速 |
| `?fade=0` | 关闭近处淡出（便于对比） |

---

## 参考

- 原论文：Kerbl, Kopanas, Leimkühler, Drettakis. *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. SIGGRAPH 2023.
- 官方实现：https://github.com/graphdeco-inria/gaussian-splatting

本仓库为独立复刻实现，未使用官方 CUDA 扩展代码。