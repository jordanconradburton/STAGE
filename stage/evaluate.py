"""
Held-out test-set evaluation of a trained checkpoint: 3D/2D RMSE, MAE,
median, P90/P95, per-axis RMSE/MAE/bias and Z error stats, true-vs-predicted
plots, and a per-window predictions CSV (positions and errors in meters).
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .checkpoint import feature_flags_of, load_models
from .data.dataset import WindowDataset, collate_windows, edges_to_device
from .data.labels import find_labels_csv
from .data.layout import read_feature_flags
from .data.preprocessing import estimate_num_sequences, preprocess_graph_folder

# Width of the '=' separator lines in the printed report.
RULE_WIDTH = 80


@torch.no_grad()
def run_inference(encoder, pos_head, test_loader, device, estimated_batches):
    """Returns (pred_m [n, 3], true_m [n, 3], exp_ids [n], sample_mse [n]) in meters."""
    preds, trues, exp_ids, sample_losses = [], [], [], []
    for x_seq, edge_index, edge_attr, label in tqdm(test_loader, desc="Running inference", total=estimated_batches):
        x_seq = x_seq.to(device)
        pred = pos_head(encoder(x_seq, edges_to_device(edge_index, device), edges_to_device(edge_attr, device)))
        pos_gt = label["position"].to(device)
        preds.append(pred.cpu().numpy())
        trues.append(pos_gt.cpu().numpy())
        sample_losses.append(((pred - pos_gt) ** 2).mean(dim=1).cpu().numpy())
        exp_ids.extend(int(e) for e in label["exp_id"])
    return np.concatenate(preds), np.concatenate(trues), np.array(exp_ids), np.concatenate(sample_losses)


def _pos_metrics(pred_m, true_m):
    """All metrics computed directly in meters."""
    diff = pred_m - true_m  # [N, 3] meters
    errs_3d = np.linalg.norm(diff, axis=1)
    errs_2d = np.linalg.norm(diff[:, :2], axis=1)
    abs_z = np.abs(diff[:, 2])
    ax_sq = diff ** 2
    ax_abs = np.abs(diff)
    return {
        "n": int(len(errs_3d)),
        "rmse_m": float(np.sqrt(np.mean(errs_3d ** 2))),
        "mae_m": float(np.mean(errs_3d)),
        "median_m": float(np.median(errs_3d)),
        "std_m": float(np.std(errs_3d)),
        "p90_m": float(np.percentile(errs_3d, 90)),
        "p95_m": float(np.percentile(errs_3d, 95)),
        "min_m": float(errs_3d.min()),
        "max_m": float(errs_3d.max()),
        "rmse_2d_m": float(np.sqrt(np.mean(errs_2d ** 2))),
        "mae_2d_m": float(np.mean(errs_2d)),
        "median_2d_m": float(np.median(errs_2d)),
        "std_2d_m": float(np.std(errs_2d)),
        "p90_2d_m": float(np.percentile(errs_2d, 90)),
        "p95_2d_m": float(np.percentile(errs_2d, 95)),
        "z_mean_m": float(abs_z.mean()),
        "z_median_m": float(np.median(abs_z)),
        "z_p95_m": float(np.percentile(abs_z, 95)),
        "axis_rmse_m": np.sqrt(ax_sq.mean(axis=0)).tolist(),
        "axis_mae_m": ax_abs.mean(axis=0).tolist(),
        "axis_mse_m2": ax_sq.mean(axis=0).tolist(),
        "axis_bias_m": diff.mean(axis=0).tolist(),
    }


def evaluate_on_test_set(pred_m, true_m, exp_ids, sample_losses, plots_dir, metrics_dir, model_info=None):
    m = _pos_metrics(pred_m, true_m)
    avg_loss = float(sample_losses.mean())

    def _sec(title):
        print(f"\n{'=' * RULE_WIDTH}\n{title}\n{'=' * RULE_WIDTH}")

    print_header = f"STAGE — TEST SET EVALUATION  ({m['n']:,} timesteps)"
    _sec(print_header)
    print("Loss function: MSE (matches training's position loss term)")
    print(f"  Average position MSE: {avg_loss:.6f} m^2\n")
    print(f"  3D RMSE:                {m['rmse_m']:>10.2f} m")
    print(f"  3D Mean (mean dist):    {m['mae_m']:>10.2f} m")
    print(f"  3D Median:              {m['median_m']:>10.2f} m")
    print(f"  3D Std:                 {m['std_m']:>10.2f} m")
    print(f"  3D P90 / P95:           {m['p90_m']:>10.2f} m  /  {m['p95_m']:.2f} m")
    print(f"  Min / Max:              {m['min_m']:>10.2f} m  /  {m['max_m']:.2f} m")
    print(f"  2D RMSE:                {m['rmse_2d_m']:>10.2f} m")
    print(f"  2D Mean:                {m['mae_2d_m']:>10.2f} m")
    print(f"  2D Median:              {m['median_2d_m']:>10.2f} m")
    print(f"  2D Std:                 {m['std_2d_m']:>10.2f} m")
    print(f"  2D P90 / P95:           {m['p90_2d_m']:>10.2f} m  /  {m['p95_2d_m']:.2f} m")
    print(f"  Z Mean / Median / P95:  {m['z_mean_m']:>10.2f} m  /  {m['z_median_m']:.2f} m  /  {m['z_p95_m']:.2f} m")
    ax_r, ax_a, bi = m["axis_rmse_m"], m["axis_mae_m"], m["axis_bias_m"]
    print(f"  Axis RMSE (x/y/z):    {ax_r[0]:>8.2f} / {ax_r[1]:>8.2f} / {ax_r[2]:>8.2f} m")
    print(f"  Axis MAE  (x/y/z):    {ax_a[0]:>8.2f} / {ax_a[1]:>8.2f} / {ax_a[2]:>8.2f} m")
    print(f"  Axis Bias (x/y/z):    {bi[0]:>8.2f} / {bi[1]:>8.2f} / {bi[2]:>8.2f} m")

    os.makedirs(plots_dir, exist_ok=True)
    axis_labels = ["X", "Y", "Z"]
    for i in range(3):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(pred_m[:, i] - true_m[:, i], bins=30, edgecolor="black")
        plt.xlabel("Error (m)"); plt.ylabel("Frequency")
        plt.title(f"{axis_labels[i]} Error Distribution"); plt.axvline(0, color="r", ls="--")
        plt.subplot(1, 2, 2)
        plt.scatter(true_m[:, i], pred_m[:, i], alpha=0.5, s=5)
        plt.xlabel(f"True {axis_labels[i]} (m)"); plt.ylabel(f"Pred {axis_labels[i]} (m)")
        plt.title(f"{axis_labels[i]} Predicted vs True")
        lo = min(true_m[:, i].min(), pred_m[:, i].min())
        hi = max(true_m[:, i].max(), pred_m[:, i].max())
        plt.plot([lo, hi], [lo, hi], "r--")
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/axis_{axis_labels[i]}_analysis.png", dpi=300)
        plt.close()

    fig = plt.figure(figsize=(12, 10))
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.scatter(*true_m.T, s=3, alpha=0.5, c="blue", label="True")
    ax3.scatter(*pred_m.T, s=3, alpha=0.5, c="red", label="Pred")
    ax3.legend(); ax3.set_xlabel("X (m)"); ax3.set_ylabel("Y (m)"); ax3.set_zlabel("Z (m)")
    plt.title("3D True vs Predicted Drone Positions")
    plt.savefig(os.path.join(plots_dir, "3D_true_vs_pred_positions.png"), dpi=300)
    plt.close()

    os.makedirs(metrics_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    diff = pred_m - true_m
    predictions_df = pd.DataFrame({
        "experiment_id": exp_ids,
        "pred_x": pred_m[:, 0], "pred_y": pred_m[:, 1], "pred_z": pred_m[:, 2],
        "true_x": true_m[:, 0], "true_y": true_m[:, 1], "true_z": true_m[:, 2],
        "error_3d": np.linalg.norm(diff, axis=1),
        "error_2d": np.linalg.norm(diff[:, :2], axis=1),
    }).round(3)
    predictions_csv = os.path.join(metrics_dir, f"Predictions_{timestamp}.csv")
    predictions_df.to_csv(predictions_csv, index=False)

    metrics = {
        "eval_timestamp": timestamp,
        "model_info": model_info or {},
        "avg_position_mse_m2": avg_loss,
        "position_metrics": m,
    }
    summary_json = os.path.join(metrics_dir, f"stage_test_summary_{timestamp}.json")
    with open(summary_json, "w") as f:
        json.dump(metrics, f, indent=2)

    _sec("OUTPUT FILES")
    print(f"  Predictions CSV:      {predictions_csv}")
    print(f"  Summary JSON:         {summary_json}")
    print("=" * RULE_WIDTH)
    return metrics


def add_arguments(p):
    p.add_argument("checkpoint_path")
    p.add_argument("test_graphs_path", help="A held-out test folder: a combined folder or a single simulator "
                                            "shard folder (graphs/shard_NNN).")
    p.add_argument("--label_csv_path", default=None, help="Defaults to the labels CSV inside test_graphs_path.")
    p.add_argument("--run_name", default=None, help="Defaults to the checkpoint's file name.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--stride", type=int, default=1,
                   help="Timesteps between the starts of consecutive test windows. The default 1 scores "
                        "every timestep from the window_size-th on; pass the window size (12) to score "
                        "only the end points of non-overlapping windows, as in training.")
    p.add_argument("--output_dir", default="eval_output")


def evaluate_folder(encoder, pos_head, cfg, device, test_graphs_path, run_name, output_dir, stride=1,
                    batch_size=256, label_csv_path=None, checkpoint_path=None):
    """Evaluates loaded models on one test folder and writes its report under
    <output_dir>/<run_name>/. Returns (metrics, number of sensors)."""
    window_size = cfg["window_size"]
    flags = feature_flags_of(cfg)
    data_flags = read_feature_flags(test_graphs_path)
    if data_flags != flags:
        raise ValueError(f"The test data's feature flags {data_flags} differ from the checkpoint's {flags}.")

    processed_path = preprocess_graph_folder(test_graphs_path, flags)
    label_csv_path = label_csv_path or find_labels_csv(test_graphs_path)
    test_dataset = WindowDataset(processed_path, label_csv_path, window_size=window_size, stride=stride)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=5,
                             pin_memory=(device.type == "cuda"), persistent_workers=True,
                             collate_fn=collate_windows)
    estimated_batches = max(1, estimate_num_sequences(processed_path, window_size, stride) // batch_size)

    print(f"\n{'=' * RULE_WIDTH}\nSTEP 1/2: Inference on {test_graphs_path}\n{'=' * RULE_WIDTH}")
    pred_m, true_m, exp_ids, sample_losses = run_inference(encoder, pos_head, test_loader, device, estimated_batches)
    print(f"Collected {len(pred_m):,} windows ({len(np.unique(exp_ids)):,} unique experiments)")

    print(f"\n{'=' * RULE_WIDTH}\nSTEP 2/2: Position-metrics analysis\n{'=' * RULE_WIDTH}")
    model_info = {
        "checkpoint_path": checkpoint_path, "run_name": run_name, "test_graphs_path": test_graphs_path,
        "num_sensors": test_dataset.num_sensors, "embed_dim": cfg["embed_dim"], "window_size": window_size,
        "stride": stride,
    }
    metrics = evaluate_on_test_set(pred_m, true_m, exp_ids, sample_losses,
                                   plots_dir=os.path.join(output_dir, run_name, "plots"),
                                   metrics_dir=os.path.join(output_dir, run_name, "metrics"), model_info=model_info)
    return metrics, test_dataset.num_sensors


def run(args):
    run_name = args.run_name or os.path.splitext(os.path.basename(args.checkpoint_path))[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    encoder, pos_head, cfg = load_models(args.checkpoint_path, device)
    evaluate_folder(encoder, pos_head, cfg, device, args.test_graphs_path, run_name, args.output_dir,
                    stride=args.stride, batch_size=args.batch_size, label_csv_path=args.label_csv_path,
                    checkpoint_path=args.checkpoint_path)
