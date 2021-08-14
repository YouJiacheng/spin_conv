import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi

half_pi = pi / 2
doub_pi = pi * 2

# 由于nn.Conv1d的padding是both size的，自己实现一个
class CircularConv1d(nn.Module):
    def __init__(self, **kwargs):
        r"""
        required params: (same as nn.Conv1d)
        in_channels: int
        out_channels: int
        kernel_size: int
        """
        super().__init__()
        self.padding = kwargs['kernel_size'] - 1

        # 屏蔽传入的padding相关参数
        kwargs['padding'] = 0
        kwargs['padding__mode'] = 'zeros' 
        self.conv1d = nn.Conv1d(**kwargs)
    
    def forward(self, input: torch.Tensor):
        return self.conv1d(F.pad(input, (0, self.padding), mode='circular'))

def grid_reference(phi: int, theta: int, dtype: torch.dtype=torch.float32):
    r"""
        phi: 纬度离散化的格数
        theta: 经度离散化的格数
    """
    phi_start = half_pi / phi # half step size
    phi_end = pi - phi_start
    theta_start = pi / theta # half step size
    theta_end = doub_pi - theta_start
    phi_tensor = torch.linspace(start=phi_start, end=phi_end, steps=phi, dtype=dtype) # phi
    theta_tensor = torch.linspace(start=theta_start, end=theta_end, steps=theta, dtype=dtype) # theta
    z = phi_tensor.cos().view(1, phi).expand(theta, phi) # theta, phi
    r = phi_tensor.sin().view(1, phi).expand(theta, phi) # theta, phi
    x = r * theta_tensor.cos().view(theta, 1).expand(theta, phi) # theta, phi
    y = r * theta_tensor.sin().view(theta, 1).expand(theta, phi) # theta, phi
    return torch.stack((x, y, z), dim=-1) # theta, phi, 3

# grid based SpinConv
class GridSpinConv(nn.Module):
    def __init__(self, phi: int, theta: int, M: int, D: int, tau: float=10, stride=1, bias=False):
        r"""
        phi: 纬度离散化的格数
        theta: 经度离散化的格数
        M: message hidden state的维数
        D: 输出维数
        """
        super().__init__()
        self.phi = phi
        self.theta = theta
        self.M = M
        self.D = D
        self.tau = tau
        self.filter = CircularConv1d(in_channels=phi*M, out_channels=D, kernel_size=theta, stride=stride)
        self.register_buffer('grid', None) # make pylance happy
        self.grid = grid_reference(phi, theta)
        
    def forward(self):
        pass

    def grid_sample_matrix(self, neighbors_relative_pos_unit: torch.Tensor):
        r"""
        Args:
            neighbors_relative_pos_unit: 邻点的单位化相对直角坐标 shape=(N, 3)
        Returns:
            grid_sample_matrix：网格对散点的采样系数矩阵 shape=(theta, phi, N)
        """
        k = torch.reshape(self.grid, (-1, 3)) # theta*phi, 3
        coef = k @ neighbors_relative_pos_unit.T # theta*phi, N
        coef = coef.sub_(1).mul_(self.tau).exp_()
        return torch.reshape(coef, (self.theta, self.phi, -1))
