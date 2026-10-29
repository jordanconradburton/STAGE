"""
Memory-efficient (exp_id, timestamp) -> drone position lookup. The labels
CSV is parsed once into a binary cache next to it (positions, plus keys
sorted for binary search) and reused on later runs.

Single drone only: the position of the first drone in each row's
`drone_positions` list is used, and any other drones are ignored.
"""
import ast
import csv
import glob
import os
from datetime import datetime

import numpy as np

LABEL_PATTERNS = ("*_labels.csv", "combined_labels.csv", "labels.csv")


def find_labels_csv(data_dir):
    """The labels CSV in a dataset folder: `combined_labels.csv` (a combined
    folder), `labels.csv` (a simulator shard) or `*_labels.csv`."""
    for pattern in LABEL_PATTERNS:
        matches = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if len(matches) > 1:
            raise ValueError(f"Several labels CSVs match {pattern} in {data_dir}: {matches}")
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No labels CSV found in {data_dir}")


class LazyLabelLoader:
    """dict-like (exp_id, timestamp_str) -> (x, y, z) lookup in meters."""

    _TS_FMT = "%Y-%m-%d %H:%M:%S.%f"  # always 26 ASCII chars

    def __init__(self, csv_path):
        base = os.path.splitext(csv_path)[0] + "_lbl_d1"
        pos_path = base + "_pos.npy"
        meta_path = base + "_meta.npz"

        if os.path.exists(pos_path) and self._cache_is_valid(csv_path, meta_path):
            print(f"Loading label cache from {os.path.basename(meta_path)}...")
            meta = np.load(meta_path)
            self._exp_ids = meta["exp_ids"]
            self._ts_strs = meta["ts_strs"]
            self._row_map = meta["row_map"]
            self._n = len(self._exp_ids)
        else:
            print(f"Building label cache for {csv_path}...")
            self._build(csv_path, pos_path, meta_path)
        # Read into memory, not memory-mapped: on a network filesystem, another
        # run rebuilding the cache replaces this file, which kills processes on
        # other machines that still map the old one.
        self._positions = np.load(pos_path)  # [n, 1, 3]

    @staticmethod
    def _cache_is_valid(csv_path, meta_path):
        """The cache must be newer than the CSV and built from a CSV of the
        same byte size."""
        try:
            csv_stat = os.stat(csv_path)
            if csv_stat.st_mtime > os.path.getmtime(meta_path) + 1.0:
                print(f"Label cache is stale: {os.path.basename(csv_path)} changed after it was built.")
                return False
            meta = np.load(meta_path)
            if "csv_size" not in meta.files or int(meta["csv_size"]) != csv_stat.st_size:
                print(f"Label cache {os.path.basename(meta_path)} does not match the CSV's size.")
                return False
        except (OSError, ValueError) as exc:
            print(f"Label cache {os.path.basename(meta_path)} could not be read ({exc}).")
            return False
        return True

    def _build(self, csv_path, pos_path, meta_path):
        with open(csv_path) as f:
            n = sum(1 for _ in f) - 1  # minus the header

        positions = np.zeros((n, 1, 3), dtype=np.float32)
        exp_ids = np.empty(n, dtype=np.int32)
        ts_strs = np.empty(n, dtype="S26")

        with open(csv_path, newline="") as f:
            for i, row in enumerate(csv.DictReader(f)):
                exp_ids[i] = int(row["exp_id"])
                ts_strs[i] = datetime.fromisoformat(row["timestamp"]).strftime(self._TS_FMT).encode("ascii")
                first = ast.literal_eval(row["drone_positions"])[0]
                positions[i, 0] = first[:3]

        order = np.lexsort((ts_strs, exp_ids))
        self._exp_ids = exp_ids[order]
        self._ts_strs = ts_strs[order]
        self._row_map = order.astype(np.int32)
        self._n = n

        # Written to temp files and renamed into place, so a run starting
        # concurrently on the same CSV never reads a half-written cache.
        tmp = f".tmp-{os.getpid()}"
        with open(pos_path + tmp, "wb") as f:
            np.save(f, positions)
        with open(meta_path + tmp, "wb") as f:
            np.savez_compressed(f, exp_ids=self._exp_ids, ts_strs=self._ts_strs, row_map=self._row_map,
                                csv_size=np.int64(os.path.getsize(csv_path)))
        os.replace(pos_path + tmp, pos_path)
        os.replace(meta_path + tmp, meta_path)
        print(f"Saved label cache ({os.path.basename(pos_path)}, {os.path.basename(meta_path)})")

    def _find_row(self, exp_id, ts_str):
        lo = int(np.searchsorted(self._exp_ids, exp_id, side="left"))
        hi = int(np.searchsorted(self._exp_ids, exp_id, side="right"))
        hits = np.where(self._ts_strs[lo:hi] == ts_str.encode("ascii"))[0]
        return int(self._row_map[lo + hits[0]]) if len(hits) else -1

    def get(self, exp_id, ts_str):
        """The drone position at (exp_id, timestamp), or None if unlabelled."""
        row = self._find_row(exp_id, ts_str)
        return None if row < 0 else self._positions[row, 0]

    def __len__(self):
        return self._n
