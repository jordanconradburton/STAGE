"""
Position decoder: a single learned query cross-attends over the per-node
sensor embeddings through a transformer decoder stack and regresses one
(x, y, z) position per window.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as CFG


class CrossAttnPosHead(nn.Module):
    """Single-drone position decoder.

    The query's self-attention over a single token reduces to a learned
    linear map of the query. Output is (x, y, z) in meters through a Softplus,
    since region coordinates are non-negative.
    """

    def __init__(self, embed_dim=CFG.EMBED_DIM, hidden_dim=CFG.DECODER_HIDDEN_DIM,
                 n_heads=CFG.DECODER_N_HEADS, n_layers=CFG.DECODER_N_LAYERS, dropout=0.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=hidden_dim,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.position_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 3),
        )

    def forward(self, node_embs):
        """node_embs: [B, N, embed_dim]. Returns [B, 3] in meters."""
        query = self.query.unsqueeze(0).expand(node_embs.size(0), -1, -1)  # [B, 1, embed_dim]
        dec_out = self.decoder(tgt=query, memory=node_embs)  # [B, 1, embed_dim]
        return F.softplus(self.position_head(dec_out.squeeze(1)))
