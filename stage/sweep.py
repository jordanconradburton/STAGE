"""
Evaluates one checkpoint, unchanged, on several test folders and writes one
summary table with a row per folder. Typically the folders hold the same kind
of scenes observed by arrays of different sizes (data_files/sensor_sweep/),
which measures how a model trained on one array size generalizes to others.
Each folder also gets the full `stage evaluate` report.
"""
import glob
import os

import pandas as pd
import torch

from .checkpoint import load_models
from .evaluate import RULE_WIDTH, evaluate_folder


def add_arguments(p):
    p.add_argument("checkpoint_path")
    p.add_argument("test_graphs_paths", nargs="+", help="Test folders, each a dataset root, a combined "
                                                        "folder or a single shard folder.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--stride", type=int, default=1,
                   help="Timesteps between the starts of consecutive test windows (see `stage evaluate`).")
    p.add_argument("--output_dir", default="eval_output")
    p.add_argument("--summary_csv", default=None,
                   help="Defaults to <output_dir>/<checkpoint name>_sweep/summary.csv.")


def _data_folder(path):
    """A dataset root (with manifest.json and graphs/) is evaluated through
    its single shard folder; any other folder is used as given."""
    shards = os.path.join(path, "graphs")
    if os.path.exists(os.path.join(path, "manifest.json")) and os.path.isdir(shards):
        names = sorted(d for d in os.listdir(shards) if os.path.isdir(os.path.join(shards, d)))
        if len(names) != 1:
            raise ValueError(f"{path} has {len(names)} shards; pass a combined folder or one shard folder.")
        return os.path.join(shards, names[0])
    return path


def _count_sensors(folder):
    first = sorted(glob.glob(os.path.join(folder, "*.pt")))
    if not first:
        raise FileNotFoundError(f"No .pt files found in {folder}")
    graphs = torch.load(first[0], map_location="cpu", weights_only=False)
    return next(iter(graphs.values()))[0].x.size(0)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    encoder, pos_head, cfg = load_models(args.checkpoint_path, device)
    sweep_dir = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.checkpoint_path))[0] + "_sweep")

    folders = [_data_folder(p) for p in args.test_graphs_paths]
    counts = [_count_sensors(f) for f in folders]
    if len(set(counts)) != len(counts):
        raise ValueError(f"Each sensor count must appear once; the folders have {counts} sensors.")

    rows = []
    for folder, n_sensors in zip(folders, counts):
        m, _ = evaluate_folder(encoder, pos_head, cfg, device, folder, run_name=f"sensors_{n_sensors:02d}",
                               output_dir=sweep_dir, stride=args.stride, batch_size=args.batch_size,
                               checkpoint_path=args.checkpoint_path)
        rows.append({
            "sensors": n_sensors, "windows": m["position_metrics"]["n"],
            "rmse_3d_m": m["position_metrics"]["rmse_m"], "mean_3d_m": m["position_metrics"]["mae_m"],
            "median_3d_m": m["position_metrics"]["median_m"], "p90_3d_m": m["position_metrics"]["p90_m"],
            "rmse_2d_m": m["position_metrics"]["rmse_2d_m"], "mean_2d_m": m["position_metrics"]["mae_2d_m"],
            "mae_x_m": m["position_metrics"]["axis_mae_m"][0], "mae_y_m": m["position_metrics"]["axis_mae_m"][1],
            "mae_z_m": m["position_metrics"]["axis_mae_m"][2],
            "test_graphs_path": folder,
        })

    df = pd.DataFrame(rows).sort_values("sensors").round(2)
    summary_csv = args.summary_csv or os.path.join(sweep_dir, "summary.csv")
    os.makedirs(os.path.dirname(summary_csv) or ".", exist_ok=True)
    df.to_csv(summary_csv, index=False)
    print(f"\n{'=' * RULE_WIDTH}\nSWEEP SUMMARY: {args.checkpoint_path}\n{'=' * RULE_WIDTH}")
    print(df.drop(columns=["test_graphs_path"]).to_string(index=False))
    print(f"\nSummary table: {summary_csv}\nPer-folder reports: {sweep_dir}/sensors_NN/")
