import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi
from .utils import rotation_to_z, repeat_tuple
half_pi = pi / 2
doub_pi = pi * 2


class CircularConv1d(nn.Module):
    r"""
    由于nn.Conv1d的padding是both size的，自己实现一个
    """

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
        kwargs['padding_mode'] = 'zeros'
        self.conv1d = nn.Conv1d(**kwargs)

    def forward(self, input: torch.Tensor):
        return self.conv1d(F.pad(input, (0, self.padding), mode='circular'))


def grid_reference(phi: int, theta: int, dtype: torch.dtype = torch.float32):
    r"""
        phi: 纬度离散化的格数
        theta: 经度离散化的格数
    """
    phi_start = half_pi / phi  # half step size
    phi_end = pi - phi_start
    theta_start = pi / theta  # half step size
    theta_end = doub_pi - theta_start
    phi_tensor = torch.linspace(start=phi_start, end=phi_end, steps=phi, dtype=dtype)  # (phi,)
    theta_tensor = torch.linspace(start=theta_start, end=theta_end, steps=theta, dtype=dtype)  # (theta,)
    z = phi_tensor.cos().view(phi, 1).expand(phi, theta)  # (phi, theta)
    r = phi_tensor.sin().view(phi, 1).expand(phi, theta)  # (phi, theta)
    x = r * theta_tensor.cos().view(1, theta).expand(phi, theta)  # (phi, theta)
    y = r * theta_tensor.sin().view(1, theta).expand(phi, theta)  # (phi, theta)
    return torch.stack((x, y, z)).flatten(start_dim=1)  # (3, phi*theta)


class GridSpinConv(nn.Module):
    r"""
    grid based SpinConv
    """

    def __init__(self, phi: int, theta: int, M: int, D: int, tau: float = 10, stride=1, bias=False):
        r"""
        phi: 纬度离散化的格数
        theta: 经度离散化的格数
        M: message维数
        D: 输出维数
        """
        super().__init__()
        self.phi = phi
        self.theta = theta
        self.M = M
        self.D = D
        self.tau = tau
        self.filter = nn.Sequential(
            CircularConv1d(in_channels=phi*M, out_channels=D, kernel_size=theta, stride=stride),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.register_buffer('grid', None)  # make pylance happy
        self.grid = grid_reference(phi, theta)  # (3, theta*phi)

    def forward(self, target_edge_index: torch.LongTensor, edge_source_index: torch.LongTensor, edge_unit: torch.Tensor, edge_message: torch.Tensor):
        r"""
        Args:
            target_edge_index: 每一个target atom对应的入边集合的index序列，入边数量不足时用e_pad范围内的index占位，shape=(..., |V| + v_pad, cutoff_num_threshold)
            edge_source_index: 每一条边的source atom index, shape=(..., |E| + e_pad)
            edge_unit: 单位化后的边向量，shape=(..., |E| + e_pad, 3) e_pad部分应为零向量
            edge_message: shape=(..., |E| + e_pad, M) e_pad部分应初始化为零向量
        Returns:
            out: shape=(..., |E| + e_pad, D)
        """
        cutoff_num_threshold = target_edge_index.size(-1)

        per_target_edge_unit = torch.gather(
            input=edge_unit.unsqueeze(-2).expand(*repeat_tuple(-1, edge_unit.dim() - 1), cutoff_num_threshold, -1),  # (..., |E| + e_pad, cutoff_num_threshold, 3)
            dim=-3,  # dim of |E| + e_pad
            index=target_edge_index.unsqueeze(-1).expand(*repeat_tuple(-1, target_edge_index.dim()), 3)  # (..., |V| + v_pad, cutoff_num_threshold, 3)
        )  # (..., |V| + v_pad, cutoff_num_threshold, 3)

        per_st_edge_unit = torch.gather(
            input=per_target_edge_unit,  # (..., |V| + v_pad, cutoff_num_threshold, 3)
            dim=-3,  # dim of |V| + v_pad
            index=edge_source_index.unsqueeze(-1).unsqueeze(-1).
            expand(*repeat_tuple(-1, edge_source_index.dim()), cutoff_num_threshold, 3)  # (..., |E| + e_pad, cutoff_num_threshold, 3)
        )  # (..., |E| + e_pad, cutoff_num_threshold, 3)

        per_st_rotations = rotation_to_z(edge_unit)  # (..., |E| + e_pad, 3, 3)
        per_st_edge_unit_local = per_st_edge_unit @ per_st_rotations  # (..., |E| + e_pad, cutoff_num_threshold, 3)
        per_st_grid_sampe_matrix_flat = self.get_grid_sample_matrix(per_st_edge_unit_local)  # (..., |E| + e_pad, cutoff_num_threshold, phi*theta)

        per_target_edge_message_t = torch.gather(
            input=edge_message.unsqueeze(-1).expand(*repeat_tuple(-1, edge_message.dim()), cutoff_num_threshold),  # (..., |E| + e_pad, M, cutoff_num_threshold)
            dim=-3,  # dim of |E| + e_pad
            index=target_edge_index.unsqueeze(-2).expand(*repeat_tuple(-1, target_edge_index.dim() - 1), self.M, -1)  # (..., |V| + v_pad, M, cutoff_num_threshold)
        )  # (..., |V| + v_pad, M, cutoff_num_threshold)

        per_st_edge_message_t = torch.gather(
            input=per_target_edge_message_t,  # (..., |V| + v_pad, M, cutoff_num_threshold)
            dim=-3,  # dim of |V| + v_pad
            index=edge_source_index.unsqueeze(-1).unsqueeze(-1).
            expand(*repeat_tuple(-1, edge_source_index.dim()), self.M, cutoff_num_threshold)  # (..., |E| + e_pad, M, cutoff_num_threshold)
        )  # (..., |E| + e_pad, M, cutoff_num_threshold)

        per_edge_grid_flat = per_st_edge_message_t @ per_st_grid_sampe_matrix_flat  # (..., |E| + e_pad, M, phi*theta)
        per_edge_grid: torch.Tensor = per_edge_grid_flat.unflatten(-1, (self.phi, self.theta))  # (..., |E| + e_pad, M, phi, theta)
        per_edge_grid_N_M_phi_theta = per_edge_grid.flatten(end_dim=-4)  # (N, M, phi, theta)
        per_edge_grid_N_Mphi_theta = per_edge_grid_N_M_phi_theta.flatten(start_dim=1, end_dim=2)  # (N, M*phi, theta)

        out_flat: torch.Tensor = self.filter(per_edge_grid_N_Mphi_theta)  # (N, D)
        out = out_flat.unflatten(0, edge_message.shape[:-1])  # (..., D)
        return out

    def get_grid_sample_matrix(self, edge_unit: torch.Tensor):
        r"""
        Args:
            edge_unit: 单位化后的边向量 shape=(..., 3)
        Returns:
            grid_sample_matrix：网格对散点的采样系数矩阵(展平的) shape=(..., phi*theta)
        """
        grid_sample_matrix = edge_unit @ self.grid  # (..., 3) @ (3, phi*theta) -> (..., phi*theta)
        return grid_sample_matrix.sub_(1).mul_(self.tau).exp_()
