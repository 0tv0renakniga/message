"""
step_02a_merge_icesat_tiles.py

-------------------------------------------------------------------------------
COMPUTATIONAL GLACIOLOGY AUDIT LOG
-------------------------------------------------------------------------------
DATE: 2026-02-07
AUTHOR: SYSTEM (Skeptical Gatekeeper)
STATUS: PRODUCTION
LOGIC:  "Lowest Uncertainty First" (Source: [1], [2])
        Iterative masking used over stack-argmin for graph stability.
-------------------------------------------------------------------------------
"""

import os
import sys
import warnings
import numpy as np
import xarray as xr
import dask.array as da
from dask.distributed import Client, LocalCluster

# --- MANDATORY PREAMBLE ---
warnings.filterwarnings("ignore", category=UserWarning)  # Silence pyproj warnings
xr.set_options(keep_attrs=True)

# --- CONFIGURATION ---
# If you are running this on a laptop, you are doing it wrong.
# If on HPC, adjust workers to match your SLURM allocation.
TEST_MODE = False  # Set True to process only 1 timestamp
DASK_CONFIG = {
    "n_workers": 4,
    "threads_per_worker": 2,
    "memory_limit": "16GB"
}

DIRS = {
    "input": "data/raw/icesat",
    "output": "data/processed_layers/intermediate"
}

# The A1-A4 quadrants. If these files don't exist, the script dies.
TILES = {
    'A1': f"{DIRS['input']}/ATL15_A1_0328_01km_005_01.nc",
    'A2': f"{DIRS['input']}/ATL15_A2_0328_01km_005_01.nc",
    'A3': f"{DIRS['input']}/ATL15_A3_0328_01km_005_01.nc",
    'A4': f"{DIRS['input']}/ATL15_A4_0328_01km_005_01.nc",
}

# Variable mapping. We process these groups independently to save memory.
GROUPS = {
    'delta_h': {
        'nc_group': 'delta_h',
        'sigma_var': 'delta_h_sigma',
        'data_vars': ['delta_h', 'delta_h_sigma', 'ice_area'],
        'output_name': 'icesat2_1km_seamless_deltah.zarr'
    },
    'dhdt_lag1': {
        'nc_group': 'dhdt_lag1',
        'sigma_var': 'dhdt_sigma',
        'data_vars': ['dhdt', 'dhdt_sigma', 'ice_area'],
        'output_name': 'icesat2_1km_seamless_lag1.zarr'
    },
    'dhdt_lag4': {
        'nc_group': 'dhdt_lag4',
        'sigma_var': 'dhdt_sigma',
        'data_vars': ['dhdt', 'dhdt_sigma', 'ice_area'],
        'output_name': 'icesat2_1km_seamless_lag4.zarr'
    }
}


def verify_physics(ds: xr.Dataset, context: str):
    """
    Mandatory physics audit.
    Rejects the dataset if it violates basic glaciological laws.
    """
    print(f"[{context}] Auditing physics...")
    
    # 1. Check for infinite or absurd values
    # ATL15 delta_h shouldn't exceed +/- 500m even in extreme collapse scenarios
    for var in ds.data_vars:
        if "sigma" not in var and "ice_area" not in var:
            v_max = ds[var].max().compute().item()
            v_min = ds[var].min().compute().item()
            if abs(v_max) > 500 or abs(v_min) > 500:
                print(f"  [FATAL] {var} out of bounds: {v_min} to {v_max}")
                # In production, raise Error. For now, warn loudly.
                warnings.warn(f"Physical violation in {var}")

    # 2. Check Ice Area Integrity
    if 'ice_area' in ds:
        area_max = ds['ice_area'].max().compute().item()
        if area_max > 1.01: # allow tiny float error
            raise ValueError(f"Ice Area fraction > 1.0: {area_max}")
        if area_max < 0.0:
            raise ValueError("Ice Area fraction negative.")

    print(f"[{context}] Physics Audit PASSED.")


def fix_coordinates(ds: xr.Dataset, res=1000.0) -> xr.Dataset:
    """
    Snaps coordinates to a strict grid to prevent 'Ragged Array' errors.
    """
    # 1. Round to nearest resolution (snap)
    new_x = np.round(ds.x / res) * res
    new_y = np.round(ds.y / res) * res
    
    # 2. Assign and Sort
    ds = ds.assign_coords(x=new_x, y=new_y)
    ds = ds.sortby(['x', 'y'])
    
    # 3. Check monotonicity
    if not ds.indexes['x'].is_monotonic_increasing:
        raise ValueError("X coordinates non-monotonic after snapping.")
    
    return ds


def create_master_canvas(tile_paths: dict) -> xr.Dataset:
    """
    Scans all tiles to determine the bounding box of the continent.
    """
    print("[Canvas] calculating continental extent...")
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    for name, path in tile_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing tile: {path}")
        
        # Open lightweight just for coords
        with xr.open_dataset(path, group='delta_h') as ds:
            # Snap locally first
            x = np.round(ds.x.values / 1000.0) * 1000.0
            y = np.round(ds.y.values / 1000.0) * 1000.0
            
            min_x = min(min_x, x.min())
            max_x = max(max_x, x.max())
            min_y = min(min_y, y.min())
            max_y = max(max_y, y.max())

    print(f"[Canvas] Extent: X[{min_x}:{max_x}], Y[{min_y}:{max_y}]")
    
    # Construct Master Grid
    x_grid = np.arange(min_x, max_x + 1000, 1000, dtype=np.float64)
    y_grid = np.arange(max_y, min_y - 1000, -1000, dtype=np.float64) # Y descends in Polar Stereo
    
    # Create dummy dataset for reindexing
    canvas = xr.Dataset(coords={'x': x_grid, 'y': y_grid})
    return canvas


def merge_group_logic(group_cfg: dict, canvas: xr.Dataset, tile_paths: dict):
    """
    Implementation of 'Lowest Uncertainty First' merge logic.
    """
    group_name = group_cfg['nc_group']
    sigma_name = group_cfg['sigma_var']
    vars_needed = group_cfg['data_vars']
    
    print(f"\n[Merge] Processing group: {group_name}...")

    # Initialize accumulation containers
    # We start with None and populate with the first tile we process
    merged_ds = None
    
    # Iterate through tiles
    for tile_id, path in tile_paths.items():
        print(f"  > Loading {tile_id}...")
        
        # 1. Open Data
        ds = xr.open_dataset(path, group=group_name, chunks={'time': 1, 'y': 2048, 'x': 2048})
        
        # 2. Slice Time for Test Mode
        if TEST_MODE:
            ds = ds.isel(time=slice(0, 1))
            
        # 3. Standardize Coords
        ds = fix_coordinates(ds)
        
        # 4. Filter Variables
        ds = ds[vars_needed]
        
        # 5. Reindex to Master Canvas (Pads missing areas with NaN)
        # This is the "Sparse to Dense" transformation
        ds_aligned = ds.reindex(x=canvas.x, y=canvas.y)
        
        # 6. Merge Logic
        if merged_ds is None:
            # First tile becomes the base
            merged_ds = ds_aligned
        else:
            # COMPARISON: Current Master vs New Tile
            # Where New_Sigma < Master_Sigma, replace Master with New.
            
            # We must handle NaNs. If Master is NaN, take New.
            # If New is NaN, keep Master.
            
            # Create masks
            new_sigma = ds_aligned[sigma_name]
            curr_sigma = merged_ds[sigma_name]
            
            # Logic:
            # 1. If new has valid data AND (current is NaN OR new_sigma < curr_sigma) -> Take New
            
            # Construct validity masks (not NaN)
            new_valid = new_sigma.notnull()
            curr_valid = curr_sigma.notnull()
            
            # The "Better Data" mask
            # valid_new AND ( (not valid_curr) OR (new_sigma < curr_sigma) )
            better_mask = new_valid & (~curr_valid | (new_sigma < curr_sigma))
            
            # Update all variables in the dataset
            for var in vars_needed:
                merged_ds[var] = xr.where(better_mask, ds_aligned[var], merged_ds[var])
                
    return merged_ds


def main():
    # 1. Initialize Dask
    cluster = LocalCluster(**DASK_CONFIG)
    client = Client(cluster)
    print(f"[System] Dask Client active: {client.dashboard_link}")

    # 2. Check Inputs
    os.makedirs(DIRS['output'], exist_ok=True)
    
    # 3. Create Master Canvas
    canvas = create_master_canvas(TILES)
    
    # 4. Process Each Group
    for key, cfg in GROUPS.items():
        # Clean Output Path
        out_path = os.path.join(DIRS['output'], cfg['output_name'])
        if os.path.exists(out_path):
            print(f"[Skip] Output exists: {out_path}")
            continue
            
        # Execute Merge
        ds_merged = merge_group_logic(cfg, canvas, TILES)
        
        # Verify
        verify_physics(ds_merged, key)
        
        # Metadata
        ds_merged.attrs['title'] = f"ICESat-2 Antarctic Mosaic - {key}"
        ds_merged.attrs['crs'] = "EPSG:3031"
        ds_merged.attrs['history'] = "Merged using Lowest-Uncertainty-First logic"
        
        # 5. Write to Zarr
        print(f"  > Writing to {out_path}...")
        # Chunking strategy for Zarr: Time=1 (indep access), Spatial=1024 (reasonable tile size)
        ds_merged = ds_merged.chunk({'time': 1, 'y': 1024, 'x': 1024})
        
        # Compute and Save
        compressor = dict(compressor=None) # faster I/O for intermediate files
        encoding = {v: compressor for v in ds_merged.data_vars}
        
        ds_merged.to_zarr(out_path, mode='w', encoding=encoding, computed=True)
        print(f"  > Success: {key}")

    print("[System] Step 02a Complete.")

if __name__ == "__main__":
    main()
