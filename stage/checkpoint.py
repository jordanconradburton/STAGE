"""Saving and loading trained encoder + position decoder checkpoints."""
import torch

from .data.layout import DEFAULT_FEATURE_FLAGS
from .model import CrossAttnPosHead, STAGEEncoder

FEATURE_FLAG_KEYS = tuple(DEFAULT_FEATURE_FLAGS)


def save_checkpoint(path, encoder, pos_head, num_features, feature_flags, window_size, train_args):
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "pos_head_state_dict": pos_head.state_dict(),
        "config": {
            "num_features": num_features,
            "hidden_dim": encoder.input_proj.out_features,
            "embed_dim": encoder.embed_dim,
            "n_spatial_layers": len(encoder.spatial_layers),
            "n_temporal_layers": len(encoder.temporal_layers),
            "decoder_hidden_dim": pos_head.position_head[1].out_features,
            "window_size": window_size,
            **feature_flags,
        },
        "train_args": train_args,
    }, path)


def load_models(checkpoint_path, device):
    """Rebuilds the encoder and position decoder from a checkpoint's own
    config. Returns (encoder, pos_head, cfg), both models in eval mode."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    encoder = STAGEEncoder(
        num_features=cfg["num_features"], hidden_dim=cfg["hidden_dim"], embed_dim=cfg["embed_dim"],
        n_spatial_layers=cfg["n_spatial_layers"], n_temporal_layers=cfg["n_temporal_layers"],
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()

    pos_head = CrossAttnPosHead(cfg["embed_dim"], hidden_dim=cfg["decoder_hidden_dim"]).to(device)
    pos_head.load_state_dict(ckpt["pos_head_state_dict"])
    pos_head.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    return encoder, pos_head, cfg


def feature_flags_of(cfg):
    return {k: bool(cfg[k]) for k in FEATURE_FLAG_KEYS}
