import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import DistanceBlock, EmbeddingBlock
from .spin_conv import GridSpinConv

from .utils import repeat_tuple


class SpinConvNet(nn.Module):
    def __init__(self,
                 phi: int,
                 theta: int,
                 M: int,
                 D: int,
                 B_i: int,
                 B_m1: int,
                 B_m2: int,
                 B_e: int,
                 K: int,
                 atom_types: int,
                 delta: float,
                 sigma: float
                 ):
        r"""
        phi: spin conv纬度离散化的格数
        theta: spin conv经度离散化的格数
        M: message维数
        D: spin conv & distance block维数
        B_i: init embedding block B
        B_m1: message embedding block 1 B
        B_m2: message embedding block 2 B
        B_e: energy embedding block B 
        K: message update轮数
        atom_types: 原子类型数
        delta: cutoff max distance
        sigma: distance gaussian basis sigma
        """
        super().__init__()
        self.spin_sonv = GridSpinConv(phi, theta, M, D)
        self.distance_block = DistanceBlock(delta, D, sigma, edge_types=atom_types*atom_types+atom_types) # 有atom_types种实际不存在的边
        self.init_fc = nn.Linear(D, D)
        self.message_fc = nn.Linear(D, D)

        self.init_emb = EmbeddingBlock(atom_embed_dim=atom_types*2, input_dim=D, B=B_i, output_dim=M)
        self.message_emb_1 = EmbeddingBlock(atom_embed_dim=atom_types*2, input_dim=D, B=B_m1, output_dim=D)
        self.message_emb_2 = EmbeddingBlock(atom_embed_dim=atom_types*2, input_dim=D, B=B_m2, output_dim=M)
        self.energy_emb = EmbeddingBlock(atom_embed_dim=atom_types, input_dim=M, B=B_e, output_dim=1)
        self.K = K
        self.atom_types = atom_types

    def forward(self,
                target_edge_index: torch.LongTensor,
                edge_source_index: torch.LongTensor,
                edge: torch.Tensor,
                source_type: torch.LongTensor,
                target_type: torch.LongTensor,
                ):
        r"""
        target_edge_index: 每一个target atom对应的入边集合的index序列，入边数量不足时用e_pad范围内的index占位，shape=(..., |V| + v_pad, cutoff_num_threshold)
        edge_source_index: 每一条边的source atom index, shape=(..., |E| + e_pad)
        edge: 边向量 (..., |E| + e_pad, 3)
        source_type: 边起点原子类型 (..., |E| + e_pad)
        target_type: 边终点原子类型 (..., |E| + e_pad)
        """
        eps = 1e-12
        edge_distance: torch.Tensor = torch.linalg.norm(edge, dim=-1, keepdim=True)  # (..., |E| + e_pad, 1)
        edge_unit = edge / (edge_distance + eps)  # (..., |E| + e_pad, 3)
        edge_type = source_type + target_type * self.atom_types
        distance_repr_raw = self.distance_block(edge_type, edge_distance)  # (..., |E| + e_pad, D)
        distance_repr_init = self.init_fc(distance_repr_raw)  # (..., |E| + e_pad, D)
        distance_repr_message = self.message_fc(distance_repr_raw)  # (..., |E| + e_pad, D)
        edge_atom_embed = torch.cat(
            (
                F.one_hot(source_type, self.atom_types),
                F.one_hot(target_type, self.atom_types),
            ),
            -1
        )
        message_init = self.init_emb(edge_atom_embed, distance_repr_init)  # (..., |E| + e_pad, M)
        message: torch.Tensor = message_init

        # K iteration of message update
        for _ in range(self.K):
            spin_conv_raw = self.spin_sonv(
                target_edge_index,
                edge_source_index,
                edge_unit,
                message
            )  # (..., |E| + e_pad, D)
            spin_conv_emb = self.message_emb_1(edge_atom_embed, spin_conv_raw)  # (..., |E| + e_pad, D)
            residual = self.message_emb_2(edge_atom_embed, spin_conv_emb + distance_repr_message)  # (..., |E| + e_pad, M)
            message += residual  # message update

        cutoff_num_threshold = target_edge_index.size(-1)
        per_target_message = torch.gather(
            input=message.unsqueeze(-2).expand(*repeat_tuple(-1, message.dim() - 1), cutoff_num_threshold, -1),  # (..., |E| + e_pad, cutoff_num_threshold, M)
            dim=-3,  # dim of |E| + e_pad
            index=target_edge_index.unsqueeze(-1).expand(*repeat_tuple(-1, target_edge_index.dim()), message.size(-1))  # (..., |V| + v_pad, cutoff_num_threshold, M)
        )  # (..., |V| + v_pad, cutoff_num_threshold, M)
        target_message_aggr = per_target_message.sum(dim=-2, keepdim=False) # (..., |V| + v_pad, M)
        per_target_energy: torch.Tensor = self.energy_emb(F.one_hot(target_type, self.atom_types), target_message_aggr) # (..., |V| + v_pad, 1)
        energy = per_target_energy.squeeze(-1).sum(-1)
        return energy