"""
Trains STAGE Encoder + Decoder on drone positions, with an
auxiliary per-node reconstruction loss (each sensor's theta sector power at
the window's last timestep, predicted from its embedding) and DropNode
augmentation.

Takes a dataset root, whose shards are combined into `<root>/combined` on
the first run (see `stage combine`), or a combined folder. Splits its
experiments 80/20 into train/validation by a stable hash of exp_id, and
saves the final-epoch model.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import save_checkpoint
from .data.combine import ensure_combined
from .data.dataset import WindowDataset, collate_windows, edges_to_device
from .data.labels import find_labels_csv
from .data.layout import N_SECTORS, read_feature_flags
from .data.preprocessing import estimate_num_sequences, preprocess_graph_folder, recon_target_slice
from .model import CrossAttnPosHead, STAGEEncoder, apply_graphgym_style_init

# The position loss is computed in kilometers.
# Reported metrics are in meters.
POSITION_LOSS_SCALE = 0.001

# Weight of the auxiliary reconstruction loss relative to the position loss.
AUX_RECON_WEIGHT = 0.1

# Weight of the non-detection term in reconstruction_loss.
SILENCE_WEIGHT = 0.3

# The GPU-resident batch cache stops growing once free GPU memory would drop
# below the larger of these.
CACHE_SAFETY_MARGIN_MIN_BYTES = 2 * 1024 ** 3
CACHE_SAFETY_MARGIN_FRACTION = 0.15

# Width of the '=' separator lines around the training banner.
RULE_WIDTH = 70


def reconstruction_loss(x_pred, x_target, silence_weight=SILENCE_WEIGHT):
    """Supervises detected (nonzero) and non-detected (zero) readings with
    separately normalized terms. Plain MSE lets the model predict zero
    everywhere (most readings are zero); supervising only nonzero targets
    lets it collapse to a constant through the output bias."""
    active = (x_target != 0).float()
    inactive = (x_target == 0).float()
    loss_act = ((x_pred - x_target).pow(2) * active).sum() / active.sum().clamp(min=1)
    loss_sil = (x_pred.pow(2) * inactive).sum() / inactive.sum().clamp(min=1)
    return loss_act + silence_weight * loss_sil


def apply_dropnode(x_seq, rate):
    """Zeroes each node's whole feature block (all timesteps) with
    probability `rate`, per sample. Returns a new tensor."""
    if rate <= 0.0:
        return x_seq
    drop = torch.rand(x_seq.shape[0], x_seq.shape[1], device=x_seq.device) < rate
    if not drop.any():
        return x_seq
    out = x_seq.clone()
    out[drop] = 0.0
    return out


def cache_still_safe(device):
    """Whether another batch can be added to the GPU-resident cache."""
    if device.type != "cuda":
        return True
    free_b, total_b = torch.cuda.mem_get_info()
    return free_b > max(CACHE_SAFETY_MARGIN_MIN_BYTES, CACHE_SAFETY_MARGIN_FRACTION * total_b)


def add_arguments(p):
    p.add_argument("train_graphs_path", help="A training dataset root (with manifest.json and graphs/), whose "
                                             "shards are combined into <root>/combined on the first run, or a "
                                             "combined folder.")
    p.add_argument("--label_csv_path", default=None,
                   help="Defaults to the combined folder's combined_labels.csv.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--window_size", type=int, default=12)
    p.add_argument("--stride", type=int, default=None,
                   help="Timesteps between the starts of consecutive training and validation windows. "
                        "Defaults to window_size (non-overlapping windows), the recipe of the paper.")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--dropnode_rate", type=float, default=0.2,
                   help="Per-node, per-sample probability of zeroing a node's features (training only).")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--checkpoint_out", default="checkpoint.pt")


def run(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_graphs_path = ensure_combined(args.train_graphs_path)
    flags = read_feature_flags(train_graphs_path)
    processed_path = preprocess_graph_folder(train_graphs_path, flags)
    label_csv_path = args.label_csv_path or find_labels_csv(train_graphs_path)

    stride = args.stride or args.window_size
    train_dataset = WindowDataset(processed_path, label_csv_path, window_size=args.window_size, stride=stride,
                                  split="train")
    val_dataset = WindowDataset(processed_path, label_csv_path, window_size=args.window_size, stride=stride,
                                split="validation",
                                labels=train_dataset.labels)

    estimated_seqs = estimate_num_sequences(processed_path, args.window_size, stride)
    estimated_train_batches = max(1, int(0.8 * estimated_seqs) // args.batch_size)
    estimated_val_batches = max(1, int(0.2 * estimated_seqs) // args.batch_size)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4,
                              pin_memory=pin, persistent_workers=True, collate_fn=collate_windows)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2,
                            pin_memory=pin, persistent_workers=True, collate_fn=collate_windows)

    encoder = STAGEEncoder(num_features=train_dataset.num_features, dropout=args.dropout).to(device)
    apply_graphgym_style_init(encoder)
    pos_head = CrossAttnPosHead(dropout=args.dropout).to(device)
    apply_graphgym_style_init(pos_head)
    recon_head = nn.Linear(encoder.embed_dim, N_SECTORS).to(device)
    apply_graphgym_style_init(recon_head)

    recon_slice = recon_target_slice(flags)

    params = list(pos_head.parameters()) + list(encoder.parameters()) + list(recon_head.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    def forward_loss(batch, training):
        x_seq, edge_index, edge_attr, label = batch
        x_seq = x_seq.to(device)
        edge_index, edge_attr = edges_to_device(edge_index, device), edges_to_device(edge_attr, device)
        pos_gt = label["position"].to(device)

        encoder_input = apply_dropnode(x_seq, args.dropnode_rate) if training else x_seq
        node_embs = encoder(encoder_input, edge_index, edge_attr)
        pred = pos_head(node_embs)

        loss = F.mse_loss(pred * POSITION_LOSS_SCALE, pos_gt * POSITION_LOSS_SCALE)
        loss_recon = reconstruction_loss(recon_head(node_embs), x_seq[:, :, recon_slice, -1])
        return loss + AUX_RECON_WEIGHT * loss_recon, pred, pos_gt, loss_recon

    # The windowed data never changes between epochs, so batches are kept on
    # the GPU after the first epoch instead of re-streamed from disk. The
    # cache is filled during epoch 1's own passes and dropped (falling back to
    # streaming) if GPU memory runs low.
    caches = {"train": [], "val": []}
    cache_enabled = {"train": True, "val": True}

    def batches(name, loader, desc, total):
        building = cache_enabled[name] and not caches[name]
        source = loader if (building or not cache_enabled[name]) else caches[name]
        for batch in tqdm(source, desc=desc, total=total, unit="batch", dynamic_ncols=True):
            if building and cache_enabled[name]:
                if cache_still_safe(device):
                    x_seq, edge_index, edge_attr, label = batch
                    batch = (x_seq.to(device), edges_to_device(edge_index, device),
                             edges_to_device(edge_attr, device), label)
                    caches[name].append(batch)
                else:
                    cache_enabled[name] = False
                    caches[name].clear()
                    torch.cuda.empty_cache()
                    print(f"GPU-resident {name} cache disabled (free GPU memory below the safety margin); "
                          f"streaming from disk from now on.")
            yield batch

    print(f"\n{'=' * RULE_WIDTH}\nTraining for {args.epochs} epochs, batch_size={args.batch_size}, "
          f"dataset={train_graphs_path}\n{'=' * RULE_WIDTH}")

    best_val_rmse = float("inf")
    for epoch in range(args.epochs):
        encoder.train(); pos_head.train(); recon_head.train()
        pbar = batches("train", train_loader, f"Train {epoch + 1}/{args.epochs}", estimated_train_batches)
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, loss_recon = forward_loss(batch, training=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

        encoder.eval(); pos_head.eval(); recon_head.eval()
        axis_sq_errs, axis_abs_errs, recon_losses = [], [], []
        with torch.no_grad():
            for batch in batches("val", val_loader, f"Val {epoch + 1}/{args.epochs}", estimated_val_batches):
                _, pred, pos_gt, loss_recon = forward_loss(batch, training=False)
                axis_sq_errs.append((pred - pos_gt) ** 2)
                axis_abs_errs.append((pred - pos_gt).abs())
                recon_losses.append(loss_recon.item())

        axis_sq = torch.cat(axis_sq_errs, dim=0)  # [n, 3]
        sq_errs = axis_sq.sum(dim=-1)
        val_rmse = torch.sqrt(sq_errs.mean()).item()
        axis_rmse = torch.sqrt(axis_sq.mean(dim=0)).tolist()
        axis_mae = torch.cat(axis_abs_errs, dim=0).mean(dim=0).tolist()
        mean_dist_m = torch.sqrt(sq_errs).mean().item()
        val_recon_loss = sum(recon_losses) / max(len(recon_losses), 1)
        best_val_rmse = min(best_val_rmse, val_rmse)

        print(f"epoch {epoch + 1}/{args.epochs}  val_rmse_m={val_rmse:.2f}  "
              f"axis_rmse_m=(x={axis_rmse[0]:.2f}, y={axis_rmse[1]:.2f}, z={axis_rmse[2]:.2f})  "
              f"axis_mae_m=(x={axis_mae[0]:.2f}, y={axis_mae[1]:.2f}, z={axis_mae[2]:.2f})  "
              f"val_mean_dist_m={mean_dist_m:.2f}  val_aux_recon_loss={val_recon_loss:.3e}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # No scheduler step after the last epoch: with T_max == epochs, that
        # step would land on eta_min = 0.
        if epoch < args.epochs - 1:
            lr_scheduler.step()

    print(f"\nBest val RMSE over epochs: {best_val_rmse:.2f} m (the saved model is the final epoch's)")

    os.makedirs(os.path.dirname(args.checkpoint_out) or ".", exist_ok=True)
    save_checkpoint(args.checkpoint_out, encoder, pos_head, train_dataset.num_features, flags,
                    args.window_size, vars(args))
    print(f"Saved checkpoint: {args.checkpoint_out}")
