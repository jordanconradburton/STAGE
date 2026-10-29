"""
Feature normalization of a folder of graph shards. The normalized shards are
written once to a `preprocessed/` subfolder next to the source and reused on
later runs.
"""
import glob
import json
import os
import shutil
from collections import defaultdict

import torch
from tqdm import tqdm

from .layout import N_SECTORS, blocks, check_width, raw_width

# Frequency bands in use, as (low, high) Hz. Up to MAX_BANDS bands are
# possible; the released data uses a single band: every emitter and sensor is
# on 433-435 MHz.
FREQUENCY_BANDS_HZ = [(433e6, 435e6)]

# The band feature is the 1-based band index divided by the number of
# possible bands, so it lies in (0, 1]. With one band in use it is the
# constant 1/6 for every sensor.
MAX_BANDS = 6

# Fixed normalization ranges, the ones the published model was trained with:
# sensor x/y in [0, 5000] m (the region size) and sensor z in [0, 100] m (the
# sensor height). Drone positions are labels, not features, and stay in raw
# meters. Data generated with a different region or sensor height is
# normalized with these same divisors.
REGION_SIZE_M = 5000.0
SENSOR_MAX_HEIGHT_M = 100.0

SOURCE_RECORD = "source.json"


def _norm_power_block(x, x_norm, block):
    """dBm -> (p+100)/60 for nonzero entries; 0 (no detection) stays 0."""
    pwr = x[:, block]
    mask = pwr != 0
    out = torch.zeros_like(pwr)
    out[mask] = (pwr[mask] + 100) / 60.0
    x_norm[:, block] = out


def normalize_features(x, flags):
    """
    Normalizes raw node features x: [N, F_raw] -> [N, F_raw - 1] (fmax is
    dropped). `flags` says which optional blocks x contains (see layout.py).

    Ranges: sensitivity [-100, -80] dBm, SINR [5, 15] dB, gain [-5, 5] dBi,
    received power [-100, 0] dBm (0 means no detection and stays 0), sector
    angle centers in degrees.
    """
    x_norm = x.clone()

    sens_idx, sinr_idx, gain_idx = 1, 2, 3
    fmin_idx, fmax_idx = 4, 5
    scount_idx = 6
    x_idx, y_idx, z_idx = 7, 8, 9

    x_norm[:, sens_idx] = (x[:, sens_idx] + 100) / 20.0
    x_norm[:, sinr_idx] = (x[:, sinr_idx] - 5) / 10.0
    x_norm[:, gain_idx] = (x[:, gain_idx] + 5) / 10.0
    x_norm[:, scount_idx] = x[:, scount_idx] / N_SECTORS
    x_norm[:, x_idx] = x[:, x_idx] / REGION_SIZE_M
    x_norm[:, y_idx] = x[:, y_idx] / REGION_SIZE_M
    x_norm[:, z_idx] = x[:, z_idx] / SENSOR_MAX_HEIGHT_M

    # fmin is replaced by the band feature (1-based band index / MAX_BANDS); fmax is dropped below.
    fmin_vals = x[:, fmin_idx]
    fmax_vals = x[:, fmax_idx]
    band_ids = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    for band_num, (lo, hi) in enumerate(FREQUENCY_BANDS_HZ):
        in_band = ((fmin_vals >= lo) & (fmin_vals <= hi)) | \
                  ((fmax_vals >= lo) & (fmax_vals <= hi))
        band_ids[in_band] = (band_num + 1) / MAX_BANDS
    x_norm[:, fmin_idx] = band_ids

    for name, block in blocks(flags).items():
        if name.endswith("_angles"):
            x_norm[:, block] = x[:, block] / 360.0
        else:
            _norm_power_block(x, x_norm, block)

    return torch.cat([x_norm[:, :fmax_idx], x_norm[:, fmax_idx + 1:]], dim=1)


def recon_target_slice(flags):
    """Slice of the theta sector-power block within a preprocessed x (fmax
    dropped, so one column left of its raw position)."""
    raw = blocks(flags)["theta_power"]
    return slice(raw.start - 1, raw.stop - 1)


def _source_record(shard_paths, flags):
    return {
        "feature_flags": flags,
        "shard_sizes": {os.path.basename(p): os.path.getsize(p) for p in shard_paths},
    }


def _check_existing(out_dir, shard_paths, flags):
    """Refuses to reuse a preprocessed folder that no longer matches its
    source shards or was built with different feature flags."""
    record_path = os.path.join(out_dir, SOURCE_RECORD)
    if os.path.exists(record_path):
        with open(record_path) as f:
            recorded = json.load(f)
        if recorded != _source_record(shard_paths, flags):
            raise ValueError(f"{out_dir} is stale: its source shards or feature flags changed since "
                             f"it was built. Delete it to rebuild.")
        return
    # Built without a source record: check the shard names and the width.
    names = sorted(os.path.basename(p) for p in shard_paths)
    built = sorted(os.path.basename(p) for p in glob.glob(os.path.join(out_dir, "*.pt")))
    if names != built:
        raise ValueError(f"{out_dir} does not hold the same shards as its source folder. "
                         f"Delete it to rebuild.")
    first = torch.load(os.path.join(out_dir, built[0]), map_location="cpu", weights_only=False)
    width = next(iter(first.values()))[0].x.shape[1]
    if width != raw_width(flags) - 1:
        raise ValueError(f"{out_dir} holds {width}-wide features, expected {raw_width(flags) - 1}. "
                         f"Delete it to rebuild.")


def preprocess_graph_folder(folder_path, flags):
    """
    Normalizes every .pt shard in a folder into `<folder>/preprocessed/`
    (built once, then reused after a staleness check). Returns that path.
    """
    shard_paths = sorted(glob.glob(os.path.join(folder_path, "*.pt")))
    if not shard_paths:
        raise FileNotFoundError(f"No .pt files found in {folder_path}")

    out_dir = os.path.join(os.path.abspath(folder_path), "preprocessed")
    if os.path.exists(out_dir):
        _check_existing(out_dir, shard_paths, flags)
        print(f"Using existing preprocessed shards: {out_dir}")
        return out_dir

    # Written into a per-process temp dir and renamed into place at the end,
    # so concurrent runs on the same folder never see a half-written folder.
    tmp_dir = f"{out_dir}.tmp-{os.getpid()}"
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"Saving preprocessed shards to: {tmp_dir} (renamed to {out_dir} when complete)")

    for shard_path in tqdm(shard_paths, desc="Preprocessing shards"):
        graphs = torch.load(shard_path, weights_only=False)

        # Grouped by node count so each group is normalized in one call.
        flat_graphs = [g for gl in graphs.values() for g in gl]
        groups = defaultdict(list)
        for i, g in enumerate(flat_graphs):
            groups[g.x.shape[0]].append((i, g))

        x_norms = [None] * len(flat_graphs)
        for n_nodes, items in groups.items():
            idxs = [i for i, _ in items]
            gs = [g for _, g in items]
            F_raw = gs[0].x.shape[1]
            check_width(F_raw, flags, shard_path)
            batch_x = torch.stack([g.x for g in gs]).view(-1, F_raw)
            batch_norm = normalize_features(batch_x, flags).view(len(gs), n_nodes, -1)
            for j, i in enumerate(idxs):
                x_norms[i] = batch_norm[j]

        for i, g in enumerate(flat_graphs):
            g.x = x_norms[i]

        torch.save(graphs, os.path.join(tmp_dir, os.path.basename(shard_path)))

    with open(os.path.join(tmp_dir, SOURCE_RECORD), "w") as f:
        json.dump(_source_record(shard_paths, flags), f, indent=2)

    try:
        os.rename(tmp_dir, out_dir)
        print(f"All preprocessed shards saved to: {out_dir}")
    except OSError:
        if os.path.exists(out_dir):
            print(f"Another process finished preprocessing {out_dir} first; discarding {tmp_dir}")
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            raise
    return out_dir


def estimate_num_sequences(graphs_path, window_size, stride):
    """Number of windows a folder yields, for progress-bar totals."""
    total = 0
    for shard_path in tqdm(sorted(glob.glob(os.path.join(graphs_path, "*.pt"))), desc="Counting windows"):
        shard_data = torch.load(shard_path, map_location="cpu", weights_only=False)
        total += sum(max(0, (len(graphs) - window_size) // stride + 1) for graphs in shard_data.values())
        del shard_data
    return total
