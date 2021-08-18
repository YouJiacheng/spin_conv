import torch
import torch.nn as nn

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
            edge_type: 边类型向量 shape=(...)
            edge_distance: 边长度向量 shape=(..., 1)
        Returns:
            shape=(..., D)
        """
        gain, offset = self.gain(edge_type), self.offset(edge_type) # (..., 1), (..., 1)
        d: torch.Tensor = gain * edge_distance + offset - self.mu # (..., D)
        return d.div_(self.sigma).square_().exp_()

class EmbeddingBlock(nn.Module):
    def __init__(self, atom_embed_dim, input_dim, B, output_dim):
        super().__init__()
        self.weighting = nn.Sequential(
            nn.Linear(atom_embed_dim, B),
            nn.Softmax(dim=-1),
            nn.Unflatten(-1, (1, B))
        )
        self.multiExpert = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, B * output_dim),
            nn.Unflatten(-1, (B, output_dim))
        )
        self.fc = nn.Sequential(
            nn.Flatten(-2, -1),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, atom_embed, input):
        r"""
        Args:
            atom_embed: 用于边时是source和target的onehot embedding concat, shape=(..., atom_embed_dim)
            input: shape=(..., input_dim)
        Returns:
            shape=(..., output_dim)
        """
        weight = self.weighting(atom_embed)     # (..., 1, B)
        variations = self.multiExpert(input)    # (..., B, output_dim)
        mixture = weight @ variations           # (..., 1, output_dim)
        return self.fc(mixture)                 # (..., output_dim)