"""针孔相机模型。

约定：
    p_cam = R @ p_world + T
    u = fx * x / z + cx
    v = fy * y / z + cy

R 为 world->camera 的旋转，T 为相机坐标系下的平移。
"""

from dataclasses import dataclass, field

import torch


@dataclass
class Camera:
    R: torch.Tensor              # [3, 3] world -> camera
    T: torch.Tensor              # [3]    world -> camera
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    znear: float = 0.05
    zfar: float = 200.0
    image_name: str = ""
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))

    def to(self, device: torch.device | str) -> "Camera":
        dev = torch.device(device)
        return Camera(
            R=self.R.to(dev),
            T=self.T.to(dev),
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            width=self.width,
            height=self.height,
            znear=self.znear,
            zfar=self.zfar,
            image_name=self.image_name,
            device=dev,
        )

    @property
    def world_view_transform(self) -> torch.Tensor:
        """[4, 4]，满足 p_cam_h = viewmat @ p_world_h。"""
        m = torch.eye(4, dtype=self.R.dtype, device=self.R.device)
        m[:3, :3] = self.R
        m[:3, 3] = self.T
        return m

    @property
    def camera_center(self) -> torch.Tensor:
        """相机在世界坐标下的位置：-R^T T。"""
        return -(self.R.transpose(0, 1) @ self.T)

    def look_at(self, eye, target, up=(0.0, 0.0, 1.0)) -> "Camera":
        """构造朝向 target 的相机（返回新的 Camera）。

        注意：这是右手系、相机 z 轴指向视野前方、y 轴朝下的约定，
        与图像坐标 (u 向右, v 向下) 一致。
        """
        eye = torch.as_tensor(eye, dtype=torch.float64, device=self.R.device)
        target = torch.as_tensor(target, dtype=torch.float64, device=self.R.device)
        up = torch.as_tensor(up, dtype=torch.float64, device=self.R.device)

        forward = target - eye
        forward = forward / forward.norm()
        right = torch.cross(forward, up, dim=0)
        right = right / right.norm()
        down = torch.cross(forward, right, dim=0)

        R = torch.stack([right, down, forward], dim=0)  # world -> camera
        T = -(R @ eye)
        return Camera(
            R=R.to(self.R.dtype),
            T=T.to(self.R.dtype),
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            width=self.width,
            height=self.height,
            znear=self.znear,
            zfar=self.zfar,
            image_name=self.image_name,
            device=self.device,
        )


def make_camera(width: int, height: int, fov_x_deg: float, device="cuda") -> Camera:
    """按水平视场角构造一个位于原点的默认相机（未摆位）。"""
    fx = 0.5 * width / torch.tan(torch.tensor(fov_x_deg * torch.pi / 180.0 / 2.0)).item()
    return Camera(
        R=torch.eye(3, dtype=torch.float64, device=device),
        T=torch.zeros(3, dtype=torch.float64, device=device),
        fx=fx,
        fy=fx,
        cx=width / 2.0,
        cy=height / 2.0,
        width=width,
        height=height,
        device=torch.device(device),
    )
