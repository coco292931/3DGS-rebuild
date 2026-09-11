"""三阶实球谐（Spherical Harmonics）求值。

与官方 3DGS 的 computeColorFromSH 保持一致：系数按 [N, K, 3] 排布，
K = (deg+1)^2，direction 取「相机指向高斯」的单位向量，
最后统一 +0.5 偏移并 clamp 到非负。
"""

import torch

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def eval_sh(deg: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """求值球谐颜色。

    Args:
        deg: 球谐阶数，0..3。
        sh: [N, K, 3]，K >= (deg+1)^2。
        dirs: [N, 3]，单位方向向量（相机 -> 高斯）。

    Returns:
        [N, 3]，可直接作为高斯颜色。
    """
    if deg < 0 or deg > 3:
        raise ValueError(f"仅支持 0..3 阶球谐，收到 {deg}")

    result = SH_C0 * sh[:, 0]

    if deg > 0:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        result = result - SH_C1 * y * sh[:, 1] + SH_C1 * z * sh[:, 2] - SH_C1 * x * sh[:, 3]

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + SH_C2[0] * xy * sh[:, 4]
                + SH_C2[1] * yz * sh[:, 5]
                + SH_C2[2] * (2.0 * zz - xx - yy) * sh[:, 6]
                + SH_C2[3] * xz * sh[:, 7]
                + SH_C2[4] * (xx - yy) * sh[:, 8]
            )

            if deg > 2:
                result = (
                    result
                    + SH_C3[0] * y * (3.0 * xx - yy) * sh[:, 9]
                    + SH_C3[1] * xy * z * sh[:, 10]
                    + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh[:, 11]
                    + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[:, 12]
                    + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh[:, 13]
                    + SH_C3[5] * z * (xx - yy) * sh[:, 14]
                    + SH_C3[6] * x * (xx - 3.0 * yy) * sh[:, 15]
                )

    return torch.clamp(result + 0.5, min=0.0)


def sh_degree_to_rest_count(deg: int) -> int:
    """给定阶数，返回除 DC 外的高阶系数个数。"""
    return (deg + 1) ** 2 - 1
