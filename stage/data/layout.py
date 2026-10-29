"""
Which feature blocks a dataset's node features contain, and where they sit.

The uav_sim simulator lays out each graph's `Data.x` as [N_sensors, F]:

    index  block
     0-9   static: sensor_type, sensitivity, sinr_threshold, gain, freq_min,
           freq_max, sector_count, x, y, z
           theta_angle_centers   (6, optional)
           theta_sector_power    (6)
           phi_angle_centers     (6, optional, only with elevation sectors)
           phi_sector_power      (6, optional)

Which optional blocks are present is an export option of the simulator. It
is recorded in the dataset's `manifest.json` under "parameters", and the
width alone cannot tell the blocks apart (theta angles + theta power and
theta power + phi power are both 22 wide), so the layout is always read from
the manifest, never inferred from the width.

The published model was trained on the default layout (every block, 34
wide).
"""
import json
import os

N_STATIC = 10         # columns 0..9
N_SECTORS = 6         # azimuth (theta) block width; sensors with fewer sectors are zero-padded
PHI_N_SECTORS = 6     # elevation (phi) block width

MANIFEST_FILENAME = "manifest.json"

# The simulator's defaults, and the layout of the published model's training data.
DEFAULT_FEATURE_FLAGS = {
    "include_sector_angle_centers": True,
    "include_phi_sectors": True,
    "include_phi_sector_angle_centers": True,
}


def find_manifest(folder):
    """Looks for manifest.json in `folder` and up to two levels above it,
    which covers a dataset root, its `combined/` folder and a single
    `graphs/shard_NNN/` folder. Returns the path, or None."""
    path = os.path.abspath(folder)
    for _ in range(3):
        candidate = os.path.join(path, MANIFEST_FILENAME)
        if os.path.exists(candidate):
            return candidate
        path = os.path.dirname(path)
    return None


def read_feature_flags(folder):
    """The feature flags of the dataset `folder` belongs to, from its
    manifest. Without a manifest, assumes the default 34-wide layout."""
    manifest_path = find_manifest(folder)
    if manifest_path is None:
        print(f"No {MANIFEST_FILENAME} found at or above {folder}; assuming the default "
              f"{raw_width(DEFAULT_FEATURE_FLAGS)}-wide feature layout.")
        return dict(DEFAULT_FEATURE_FLAGS)
    with open(manifest_path) as f:
        params = json.load(f)["parameters"]
    return {
        "include_sector_angle_centers": bool(params["include_azimuth_angle_centers"]),
        "include_phi_sectors": bool(params["elevation_sectors"]),
        "include_phi_sector_angle_centers": bool(params["elevation_sectors"]
                                                 and params["include_elevation_angle_centers"]),
    }


def blocks(flags):
    """Column slices of each block present in a raw (unnormalized) x."""
    out = {}
    cur = N_STATIC
    if flags["include_sector_angle_centers"]:
        out["theta_angles"] = slice(cur, cur + N_SECTORS)
        cur += N_SECTORS
    out["theta_power"] = slice(cur, cur + N_SECTORS)
    cur += N_SECTORS
    if flags["include_phi_sectors"]:
        if flags["include_phi_sector_angle_centers"]:
            out["phi_angles"] = slice(cur, cur + PHI_N_SECTORS)
            cur += PHI_N_SECTORS
        out["phi_power"] = slice(cur, cur + PHI_N_SECTORS)
        cur += PHI_N_SECTORS
    return out


def raw_width(flags):
    return max(s.stop for s in blocks(flags).values())


def check_width(width, flags, where):
    expected = raw_width(flags)
    if width != expected:
        raise ValueError(
            f"{where}: node features are {width} wide, but the dataset's feature flags {flags} "
            f"describe a {expected}-wide layout.")
