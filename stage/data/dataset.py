"""
Streaming sliding-window dataset over a folder of preprocessed graph shards,
with the drone position at each window's last timestep as its label.
"""
import glob
import os
import zlib

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .labels import LazyLabelLoader


def _stable_exp_hash(exp_id):
    """crc32 rather than hash(): str hashing is randomized per process, which
    would change the train/validation split between launches."""
    return zlib.crc32(str(exp_id).encode())


class WindowDataset(IterableDataset):
    """Yields (x_seq [N, F, T], edge_index, edge_attr, label) for every
    window of `window_size` timesteps of every experiment, starting every
    `stride` timesteps.

    Each experiment has its own sensors: positions, static features and
    inter-sensor distances all differ between experiments, and every window
    carries its own experiment's node features and edges. Within an
    experiment the sensors are fixed, so the edges are read once. All
    experiments in a folder have the same sensor count, which lets windows be
    stacked into batches.

    Each window is labelled with the drone's position at its last timestep.
    With stride == window_size (the default, used for training and
    validation) windows do not overlap and an experiment of T timesteps
    yields floor(T / window_size) windows. With stride 1 (the default for
    test evaluation) every timestep from the window_size-th on gets its own
    window and prediction.

    split: None (every experiment), or "train"/"validation", an 80/20 split by
    a stable hash of exp_id.
    """

    def __init__(self, graphs_path, label_csv_path, window_size=12, stride=None, split=None, labels=None):
        super().__init__()
        if split not in (None, "train", "validation"):
            raise ValueError(f"split must be None, 'train' or 'validation', got {split!r}")
        self.window_size = window_size
        self.stride = window_size if stride is None else stride
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        self.split = split

        self.shard_paths = sorted(glob.glob(os.path.join(graphs_path, "*.pt")))
        if not self.shard_paths:
            raise FileNotFoundError(f"No .pt files found in {graphs_path}")

        first_shard = torch.load(self.shard_paths[0], map_location="cpu", weights_only=False)
        first_graph = next(iter(first_shard.values()))[0]
        self.num_features = first_graph.x.size(1)
        self.num_sensors = first_graph.x.size(0)
        del first_shard

        # `labels` is a lookup table (exp_id, timestamp) -> position for the
        # whole folder. The train and validation datasets of one folder can
        # share one loaded copy to save memory; each still looks up only its
        # own experiments (see _include).
        self.labels = labels if labels is not None else LazyLabelLoader(label_csv_path)
        print(f"{len(self.labels):,} labels")

    def _include(self, exp_id):
        if self.split is None:
            return True
        in_val = _stable_exp_hash(exp_id) % 5 == 0
        return in_val if self.split == "validation" else not in_val

    def __iter__(self):
        worker = get_worker_info()
        shard_paths = self.shard_paths if worker is None else self.shard_paths[worker.id::worker.num_workers]
        for shard_path in shard_paths:
            shard_data = torch.load(shard_path, map_location="cpu", weights_only=False)

            for exp_id, graphs in shard_data.items():
                if not self._include(exp_id) or len(graphs) < self.window_size:
                    continue

                edge_index = graphs[0].edge_index
                edge_attr = graphs[0].edge_attr
                exp_features = torch.stack([g.x for g in graphs], dim=-1)  # [N, F, T_exp]

                for start in range(0, len(graphs) - self.window_size + 1, self.stride):
                    end = start + self.window_size
                    position = self.labels.get(exp_id, str(graphs[end - 1].timestamp).strip())
                    if position is None:
                        continue
                    label = {"position": torch.tensor(np.array(position), dtype=torch.float),
                             "exp_id": exp_id}
                    yield exp_features[:, :, start:end].clone(), edge_index, edge_attr, label

            del shard_data


def collate_windows(batch):
    """Stacks node features into [B, N, F, T]; edges stay a tuple of B
    per-sample tensors."""
    x_seqs, edge_indices, edge_attrs, labels = zip(*batch)
    return (torch.stack(x_seqs, dim=0), tuple(edge_indices), tuple(edge_attrs),
            {"position": torch.stack([l["position"] for l in labels]),
             "exp_id": [l["exp_id"] for l in labels]})


def edges_to_device(edges, device):
    return tuple(e.to(device) for e in edges)
