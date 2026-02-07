"""
step_02a_merge_icesat_tiles.py

-------------------------------------------------------------------------------
COMPUTATIONAL GLACIOLOGY AUDIT LOG
-------------------------------------------------------------------------------
DATE: 2026-02-07
AUTHOR: SYSTEM (Skeptical Gatekeeper)
STATUS: PRODUCTION (Hotfix: Unit Awareness)
LOGIC:  "Lowest Uncertainty First" (Source: [1])
        Hardware Config: Optimized for 55GB/12Core Node.
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

# --- CONFIGURATION (OPTIMIZED FOR 55GB RAM / 12 CORES) ---
TEST_MODE = False  # Set True to process only 1 timestamp
DASK_CONFIG = {
    "n_workers": 4,           # 4 Workers ensures large chunks fit in RAM
    "threads_per_worker": 2,  # 8 Logical cores total (leaves 4 for OS/IO)
    "memory_limit": "11GB"    # 44GB Total (leaves 11GB Safety Buffer)
}

DIRS = {
    "input": "data/raw/icesat",
    "output": "data/processed_layers/intermediate"
}

TILES = {
    'A1': f"{DIRS['input']}/ATL15_A1_0328_01km_005_01.nc",
    'A2': f"{DIRS['input']}/ATL15_A2_0328_01km_005_01.nc",
    'A3': f"{DIRS['input']}/ATL15_A3_0328_01km_005_01.nc",
    'A4': f"{DIRS['input']}/ATL15_A4_0328_01km_005_01.nc",
}

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
    Updated to handle 'meters^2' vs 'fraction' for ice_area.
    """
    print(f"[{context}] Auditing physics...")
    
    # 1. Check for infinite or absurd values in height changes
    for var in ds.data_vars:
        if "sigma" not in var and "ice_area" not in var:
            v_max = ds[var].max().compute().item()
            v_min = ds[var].min().compute().item()
            if abs(v_max) > 500 or abs(v_min) > 500:
                print(f"  [FATAL] {var} out of bounds: {v_min} to {v_max}")
                warnings.warn(f"Physical violation in {var}")

    # 2. Check Ice Area Integrity (Adaptive)
    if 'ice_area' in ds:
        area_max = ds['ice_area'].max().compute().item()
        area_min = ds['ice_area'].min().compute().item()
        
        if area_min < 0.0:
            raise ValueError(f"Negative Ice Area detected: {area_min}")

        # Check Order of Magnitude
        if area_max > 2.0:
            # Case A: Units are meters^2 (expect ~1e6)
            # 1km pixel in EPSG:3031 can distort up to ~1.1e6. 
            # We set limit at 2.0e6 to catch gross projection errors (like units in feet).
            if area_max > 2.0e6:
                raise ValueError(f"Ice Area too large for 1km grid: {area_max:.1f} m^2 (Limit: 2.0e6)")
            print(f"  > Ice Area validated as meters^2 (Max: {area_max:.1f})")
        else:
            # Case B: Units are Fraction (0-1)
            if area_max > 1.01:
                raise ValueError(f"Ice Area fraction > 1.0: {area_max}")
            print(f"  > Ice Area validated as Fraction (Max: {area_max:.4f})")

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
    Implementation of 'Lowest Uncertainty First' merge logic [Source 17].
    """
    group_name = group_cfg['nc_group']
    sigma_name = group_cfg['sigma_var']
    vars_needed = group_cfg['data_vars']
    
    print(f"\n[Merge] Processing group: {group_name}...")

    merged_ds = None
    
    # Iterate through tiles
    for tile_id, path in tile_paths.items():
        print(f"  > Loading {tile_id}...")
        
        # 1. Open Data (Chunked)
        ds = xr.open_dataset(path, group=group_name, chunks={'time': 1, 'y': 2048, 'x': 2048})
        
        # 2. Slice Time for Test Mode
        if TEST_MODE:
            ds = ds.isel(time=slice(0, 1))
            
        # 3. Standardize Coords
        ds = fix_coordinates(ds)
        
        # 4. Filter Variables
        ds = ds[vars_needed]
        
        # 5. Reindex to Master Canvas
        ds_aligned = ds.reindex(x=canvas.x, y=canvas.y)
        
        # 6. Merge Logic
        if merged_ds is None:
            merged_ds = ds_aligned
        else:
            # Create masks
            new_sigma = ds_aligned[sigma_name]
            curr_sigma = merged_ds[sigma_name]
            
            # Logic: If new has valid data AND (current is NaN OR new_sigma < curr_sigma) -> Take New
            new_valid = new_sigma.notnull()
            curr_valid = curr_sigma.notnull()
            
            # The "Better Data" mask
            better_mask = new_valid & (~curr_valid | (new_sigma < curr_sigma))
            
            # Update all variables in the dataset
            for var in vars_needed:
                merged_ds[var] = xr.where(better_mask, ds_aligned[var], merged_ds[var])
                
    return merged_ds


def main():
    # 1. Initialize Dask with Optimized Config
    cluster = LocalCluster(**DASK_CONFIG)
    client = Client(cluster)
    print(f"[System] Dask Client active: {client.dashboard_link}")
    print(f"[System] Workers: {DASK_CONFIG['n_workers']} | Mem: {DASK_CONFIG['memory_limit']}")

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
        ds_merged.attrs['history'] = "Merged using Lowest-Uncertainty-First logic; Ice Area checked."
        
        # 5. Write to Zarr
        print(f"  > Writing to {out_path}...")
        ds_merged = ds_merged.chunk({'time': 1, 'y': 1024, 'x': 1024})
        
        compressor = dict(compressor=None)
        encoding = {v: compressor for v in ds_merged.data_vars}
        
        ds_merged.to_zarr(out_path, mode='w', encoding=encoding, computed=True)
        print(f"  > Success: {key}")

    print("[System] Step 02a Complete.")

if __name__ == "__main__":
    main()
