"""Weight initialization."""
import torch.nn as nn


def apply_graphgym_style_init(module):
    """
    Reproduction of torch_geometric.graphgym.init.init_weights: every
    nn.Linear gets Xavier/Glorot uniform weights (gain='relu') and a zero
    bias, every nn.LayerNorm/BatchNorm1d gets ones/zeros.

    This does not reach torch_geometric.nn.Linear (GATv2Conv's internal
    projections), which keep PyG's own Glorot init, as in the reference
    implementation.
    """
    gain = nn.init.calculate_gain("relu")
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    return module
