"""
Permutation-invariant, size-agnostic graph-temporal encoder for a network of
stationary ground sensors observing a single RSS-emitting target over a
sliding time window.

Each sensor node carries a per-timestep feature vector (static sensor
metadata + per-sector received power); edges carry inter-sensor distances.
STAGEEncoder runs spatial graph attention (GATv2Conv) per timestep, then
temporal self-attention across the window for each node, and returns each
node's embedding at the last timestep.
"""
import math

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

from . import config as CFG


def _sinusoidal_pe(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class STAGEEncoder(nn.Module):

    def __init__(self, num_features, hidden_dim=CFG.HIDDEN_DIM, embed_dim=CFG.EMBED_DIM, dropout=0.0,
                 n_spatial_layers=CFG.N_SPATIAL_LAYERS, n_temporal_layers=CFG.N_TEMPORAL_LAYERS):
        super().__init__()
        assert embed_dim % CFG.N_HEADS == 0, "embed_dim must be divisible by N_HEADS"
        self.embed_dim = embed_dim

        self.input_proj = nn.Linear(num_features, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        self.spatial_layers = nn.ModuleList()
        for i in range(n_spatial_layers):
            in_dim = hidden_dim if i == 0 else embed_dim
            self.spatial_layers.append(
                GATv2Conv(in_dim, embed_dim // CFG.N_HEADS, heads=CFG.N_HEADS, edge_dim=1, dropout=dropout)
            )
        self.spatial_act = nn.ReLU()
        # Applied to each spatial layer's output before the residual add,
        # independent of GATv2Conv's own attention dropout.
        self.spatial_out_dropout = nn.Dropout(dropout)

        self.register_buffer("temporal_pe", _sinusoidal_pe(CFG.MAX_WINDOW_SIZE, embed_dim))

        self.temporal_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=CFG.N_HEADS, dropout=dropout, batch_first=True)
            for _ in range(n_temporal_layers)
        ])
        self.temporal_ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Dropout(dropout),
                          nn.Linear(embed_dim * 2, embed_dim), nn.Dropout(dropout))
            for _ in range(n_temporal_layers)
        ])

    @staticmethod
    def _edge_feature(edge_attr):
        """Distance -> decay weight in [1e-6, 1]. Distances are divided by the
        graph's own largest edge distance, and exp(-d) is shifted and rescaled
        to be 1 at d = 0 and 0 at the largest distance; the result is then
        clamped to a floor of 1e-6, so the farthest pair of sensors in each
        graph gets weight 1e-6, never 0."""
        d = edge_attr.squeeze(-1) / edge_attr.max().clamp(min=1e-6)
        decay = (torch.exp(-d) - math.exp(-1.0)) / (1.0 - math.exp(-1.0))
        return decay.clamp(min=1e-6).unsqueeze(-1)

    def _spatial_forward(self, x_seq, edge_index, edge_attr, T):
        """input_proj + the GATv2Conv stack over the whole [B, T] batch as one
        block-diagonal graph. edge_index/edge_attr are tuples of B per-sample
        tensors. Returns [B, T, N, embed_dim]."""
        B, N = x_seq.shape[0], x_seq.shape[1]
        device = x_seq.device

        x_blocks, edge_blocks, edge_feat_blocks = [], [], []
        for b in range(B):
            ei_b, ea_b = edge_index[b], edge_attr[b]
            x_blocks.append(x_seq[b].permute(2, 0, 1).reshape(T * N, -1))  # [N, F, T] -> [T*N, F]
            if ei_b.numel() > 0:
                edge_blocks.append(torch.cat([ei_b + b * T * N + t * N for t in range(T)], dim=1))
                edge_feat_blocks.append(self._edge_feature(ea_b).repeat(T, 1))

        x_batched = torch.cat(x_blocks, dim=0)
        edge_index_batched = (torch.cat(edge_blocks, dim=1) if edge_blocks
                              else torch.zeros((2, 0), dtype=edge_index[0].dtype, device=device))
        edge_feat_batched = torch.cat(edge_feat_blocks, dim=0) if edge_feat_blocks else None

        h = self.input_dropout(self.input_proj(x_batched))
        for layer in self.spatial_layers:
            h_new = layer(h, edge_index_batched, edge_attr=edge_feat_batched)
            h_new = self.spatial_out_dropout(self.spatial_act(h_new))
            h = h_new + h if h.size(-1) == h_new.size(-1) else h_new

        return h.view(B, T, N, self.embed_dim)

    def _temporal_attend(self, seq):
        """seq: [B*N, T, embed_dim]; each node's timestep sequence attends
        independently."""
        seq = seq + self.temporal_pe[:seq.size(1), :].unsqueeze(0)
        for attn, ffn in zip(self.temporal_layers, self.temporal_ffns):
            attn_out, _ = attn(seq, seq, seq)
            seq = seq + attn_out
            seq = seq + ffn(seq)
        return seq

    def forward(self, x_seq, edge_index, edge_attr):
        """x_seq: [B, N, F, T]; edge_index/edge_attr: tuples of B per-sample
        tensors. Returns node embeddings [B, N, embed_dim] at the last
        timestep."""
        B, N, _, T = x_seq.shape
        if T > CFG.MAX_WINDOW_SIZE:
            raise ValueError(f"window of {T} timesteps exceeds MAX_WINDOW_SIZE={CFG.MAX_WINDOW_SIZE}")

        h = self._spatial_forward(x_seq, edge_index, edge_attr, T)
        seq = h.permute(0, 2, 1, 3).reshape(B * N, T, self.embed_dim)
        seq = self._temporal_attend(seq)
        return seq.view(B, N, T, self.embed_dim)[:, :, -1, :]
