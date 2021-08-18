import torch
import torch.nn as nn
import torch.nn.functional as F

class DistanceBlock(nn.Module):
    def __init__(self, delta, D, sigma, edge_types):
        r"""
        delta: μ_i ∈ [0, delta], delta是cutoff的最大半径
        D: 输出维数
        sigma: 标准差
        edge_types: 边类型数（原子对类型数）
        """
        super().__init__()
        self.register_buffer('mu', None) # make pylance happy
        self.mu = torch.linspace(start=0, end=delta, steps=D, dtype=torch.float32)
        self.sigma = sigma
        self.gain = nn.Embedding(num_embeddings=edge_types, embedding_dim=1)
        self.offset = nn.Embedding(num_embeddings=edge_types, embedding_dim=1)

    def forward(self, edge_type, edge_distance):
        r"""
        Args:
            edge_type: 边类型向量 shape=(N, 1)
            edge_distance: 边长度向量 shape=(N, 1)
        Returns:
            shape=(N, D)
        """
        gain, offset = self.gain(edge_type), self.offset(edge_type)
        d: torch.Tensor = gain * edge_distance + offset - self.mu # shape=(N, D)
        return d.div_(self.sigma).square_().exp_()
        