"""
Combines a dataset's per-shard folders into the single folder the model
trains on.

The simulator writes each shard to its own folder (`<root>/graphs/shard_NNN/`
with `graphs.pt` and `labels.csv`), and exp_id restarts at 1 in every shard.
This step offsets each shard's exp_ids so they are unique across the
dataset, writes every shard's graphs under the new ids, and concatenates
the label files into one `combined_labels.csv`:

    <root>/combined/
        shard_0_batch_001.pt
        shard_10_batch_001.pt
        ...
        combined_labels.csv

A simulator shard folder `shard_NNN` is named by its unpadded index
(`shard_7`, not `shard_007`), and shards are taken in sorted name order:
shard_0, shard_1, shard_10, ..., shard_19, shard_2, shard_20, ... Each
shard's offset is the sum of (max exp_id + 1) over the shards before it.
"""
import glob
import os
import re
import shutil

import pandas as pd
import torch
from tqdm import tqdm

from .layout import MANIFEST_FILENAME

COMBINED_DIRNAME = "combined"
GRAPHS_DIRNAME = "graphs"
COMBINED_LABELS = "combined_labels.csv"


def shard_name(dirname):
    """`shard_007` -> `shard_7`; any other folder name is kept as is."""
    m = re.fullmatch(r"shard_(\d+)", dirname)
    return f"shard_{int(m.group(1))}" if m else dirname


def find_shard_dirs(data_path):
    """(name, path, labels_csv) for every sub-folder of data_path holding
    both .pt shards and a labels CSV, in sorted name order (see module
    docstring)."""
    shards = []
    for dirname in sorted(os.listdir(data_path), key=shard_name):
        name = shard_name(dirname)
        path = os.path.join(data_path, dirname)
        if not os.path.isdir(path) or COMBINED_DIRNAME in name.lower():
            continue
        csvs = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not glob.glob(os.path.join(path, "*.pt")) or not csvs:
            continue
        labelled = [c for c in csvs if "label" in os.path.basename(c).lower()]
        shards.append((name, path, (labelled or csvs)[0]))
    if not shards:
        raise FileNotFoundError(f"No shard folders (with .pt files and a labels CSV) found in {data_path}")
    return shards


def combine(data_path, out_dir):
    """Writes the combined folder for the shard folders in data_path."""
    shards = find_shard_dirs(data_path)
    tmp_dir = f"{out_dir}.tmp-{os.getpid()}"
    os.makedirs(tmp_dir)

    all_labels = []
    seen = set()
    offset = 0
    for name, path, labels_csv in shards:
        df = pd.read_csv(labels_csv)
        if "exp_id" not in df.columns:
            raise ValueError(f"{labels_csv} has no exp_id column")
        shard_offset = offset
        offset += int(df["exp_id"].max()) + 1
        print(f"{name}: {df['exp_id'].nunique()} experiments, exp_id offset {shard_offset}")

        df["exp_id"] = df["exp_id"].astype(int) + shard_offset
        all_labels.append(df)

        for batch_idx, batch_path in enumerate(tqdm(sorted(glob.glob(os.path.join(path, "*.pt"))),
                                                    desc=f"  {name}")):
            graphs = torch.load(batch_path, weights_only=False)
            adjusted = {}
            for exp_id, g_list in graphs.items():
                new_id = int(exp_id) + shard_offset
                if new_id in seen:
                    raise ValueError(f"Duplicate exp_id {new_id} after offsetting {batch_path}")
                adjusted[new_id] = g_list
                seen.add(new_id)
            torch.save(adjusted, os.path.join(tmp_dir, f"{name}_batch_{batch_idx + 1:03d}.pt"))

    combined_df = pd.concat(all_labels, ignore_index=True)
    combined_df.sort_values(by="exp_id", inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    combined_df.to_csv(os.path.join(tmp_dir, COMBINED_LABELS), index=False)

    try:
        os.rename(tmp_dir, out_dir)
    except OSError:
        # Another run (e.g. a second seed started at the same time) finished
        # combining the same dataset first; its folder is identical.
        if not os.path.exists(os.path.join(out_dir, COMBINED_LABELS)):
            raise
        shutil.rmtree(tmp_dir)
        print(f"{out_dir} was completed by another run meanwhile; using it")
        return out_dir
    print(f"Combined {len(shards)} shards ({len(seen):,} experiments) into {out_dir}")
    return out_dir


def is_combined(folder):
    return os.path.exists(os.path.join(folder, COMBINED_LABELS))


def ensure_combined(dataset):
    """The combined folder to train on, combining first if needed.

    `dataset` is a combined folder (used as is) or a dataset root, which is
    combined into `<root>/combined` unless that already exists. A single
    shard folder is refused: training on it would silently use one shard of
    the dataset."""
    if not os.path.isdir(dataset):
        raise SystemExit(f"{dataset} is not a folder.")
    if is_combined(dataset):
        print(f"Using combined folder: {dataset}")
        return dataset
    if glob.glob(os.path.join(dataset, "*.pt")):
        raise SystemExit(f"{dataset} is a single shard folder. Pass the dataset root (the folder with "
                         f"manifest.json and graphs/) to train on all of its shards.")
    data_path, out_dir = resolve_paths(dataset)
    if is_combined(out_dir):
        print(f"Combined dataset already exists, skipping combination: {out_dir}")
        return out_dir
    if os.path.exists(out_dir):
        raise SystemExit(f"{out_dir} exists but has no {COMBINED_LABELS}. Delete it and rerun to rebuild it.")
    print(f"Combining the shards of {data_path} into {out_dir} (once per dataset)")
    return combine(data_path, out_dir)


def resolve_paths(dataset, out_dir=None):
    """A simulator dataset root (with manifest.json or a `graphs/` folder)
    combines its `graphs/` folder into `<root>/combined`, also when given
    the `graphs/` folder itself; any other folder of shard folders combines
    into `<folder>/combined`."""
    if not os.path.isdir(dataset):
        raise SystemExit(f"{dataset} is not a folder.")
    dataset = os.path.normpath(dataset)
    if os.path.basename(dataset) == GRAPHS_DIRNAME and os.path.exists(
            os.path.join(os.path.dirname(dataset), MANIFEST_FILENAME)):
        dataset = os.path.dirname(dataset)
    data_path = dataset
    if (os.path.exists(os.path.join(dataset, MANIFEST_FILENAME))
            or os.path.isdir(os.path.join(dataset, GRAPHS_DIRNAME))):
        data_path = os.path.join(dataset, GRAPHS_DIRNAME)
    return data_path, out_dir or os.path.join(dataset, COMBINED_DIRNAME)


def add_arguments(p):
    p.add_argument("dataset", help="A simulator dataset root (with manifest.json and graphs/), "
                                   "or any folder of shard folders.")
    p.add_argument("--out_dir", default=None, help="Defaults to <dataset>/combined.")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing combined folder.")


def run(args):
    data_path, out_dir = resolve_paths(args.dataset, args.out_dir)
    if os.path.exists(out_dir):
        if not args.overwrite:
            raise SystemExit(f"{out_dir} already exists. Pass --overwrite to rebuild it.")
        shutil.rmtree(out_dir)
    combine(data_path, out_dir)
