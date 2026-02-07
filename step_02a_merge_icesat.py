"""
step_02a_merge_icesat_tiles.py

PRINCIPAL ENGINEER NOTES:
-------------------------
1. Objective: Merge ATL15 quadrants (A1-A4) into seamless 1km intermediate products.
2. Physics: "Lowest Uncertainty First" (Source: [3]). 
   - We calculate a 'Winner Index' based on the sigma variable.
   - This index drives the selection for ALL associated variables.
3. Operations: Includes TEST_MODE for rapid validation (Source: [1]).
"""

import os
import warnings
import numpy as np
import xarray as xr
import dask.array as da
from dask.distributed import Client, LocalCluster
from pyproj import CRS

# Suppress minor geospatial warnings
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

# >>> TEST FLAG <<<
TEST_MODE = True  # Set to False for Production Run

CRS_EPSG = "EPSG:3031"
INPUT_DIR = "data/raw/icesat"
OUTPUT_DIR = "data/processed_layers/intermediate"

TILES = {
    'A1': f"{INPUT_DIR}/ATL15_A1_0328_01km_005_01.nc",
    'A2': f"{INPUT_DIR}/ATL15_A2_0328_01km_005_01.nc",
    'A3': f"{INPUT_DIR}/ATL15_A3_0328_01km_005_01.nc",
    'A4': f"{INPUT_DIR}/ATL15_A4_0328_01km_005_01.nc",
}

GROUPS = {
    'delta_h': {
        'nc_group': 'delta_h',
        'sigma_var': 'delta_h_sigma',
        'vars_to_keep': ['delta_h', 'delta_h_sigma', 'ice_area', 'data_count', 'misfit_rms'],
        'output_suffix': 'deltah'
    },
    'dhdt_lag1': {
        'nc_group': 'dhdt_lag1',
        'sigma_var': 'dhdt_sigma',
        'vars_to_keep': ['dhdt', 'dhdt_sigma', 'ice_area'],
        'output_suffix': 'lag1'
    },
    'dhdt_lag4': {
        'nc_group': 'dhdt_lag4',
        'sigma_var': 'dhdt_sigma',
        'vars_to_keep': ['dhdt', 'dhdt_sigma', 'ice_area'],
        'output_suffix': 'lag4'
    }
}

DASK_WORKERS = 4
DASK_THREADS = 2
DASK_MEMORY = "16GB"

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def fix_coordinates(ds: xr.Dataset, resolution: float = 1000.0) -> xr.Dataset:
    """Snaps coordinates to nearest 1km grid to prevent misalignment."""
    ds = ds.sortby('x').sortby('y', ascending=False)
    new_x = np.round(ds.x.values / resolution) * resolution
    new_y = np.round(ds.y.values / resolution) * resolution
    ds = ds.assign_coords(x=new_x, y=new_y)
    return ds

def preproc_tile(ds: xr.Dataset) -> xr.Dataset:
    """Decodes time and standardizes dataset."""
    if not np.issubdtype(ds.time.dtype, np.datetime64):
        if 'units' not in ds.time.attrs:
            ds.time.attrs['units'] = "days since 2018-01-01"
        ds = xr.decode_cf(ds)
    
    # >>> TEST MODE SLICING <<<
    if TEST_MODE:
        # Slice lazily - Dask will optimize the load
        ds = ds.isel(time=slice(0, 1))
        
    return ds

def create_virtual_1km_grid(datasets: list) -> xr.Dataset:
    """Creates the seamless canvas for the continent."""
    x_min = min([ds.x.min().item() for ds in datasets])
    x_max = max([ds.x.max().item() for ds in datasets])
    y_min = min([ds.y.min().item() for ds in datasets])
    y_max = max([ds.y.max().item() for ds in datasets])
    
    x_coords = np.arange(x_min, x_max + 1000, 1000)
    y_coords = np.arange(y_max, y_min - 1000, -1000)
    
    return xr.Dataset(coords={'x': x_coords, 'y': y_coords, 'time': datasets.time})

def merge_group(datasets: list, template: xr.Dataset, config: dict) -> xr.Dataset:
    """Merges a specific variable group using 'Lowest Uncertainty First'."""
    sigma_var = config['sigma_var']
    vars_to_keep = config['vars_to_keep']
    
    print(f"    [Merge] Stacking tiles for {sigma_var}...")
    
    aligned = []
    for i, ds in enumerate(datasets):
        valid_vars = [v for v in vars_to_keep if v in ds]
        ds_sub = ds[valid_vars]
        ds_aligned = ds_sub.reindex_like(template, method=None, tolerance=1e-5)
        ds_aligned['tile_id'] = i
        aligned.append(ds_aligned)
        
    combined = xr.concat(aligned, dim='tile')
    
    print(f"    [Merge] resolving overlaps via nanargmin({sigma_var})...")
    sigma_filled = combined[sigma_var].fillna(np.inf)
    best_idx = sigma_filled.argmin(dim='tile')
    
    merged = combined.isel(tile=best_idx)
    return merged.drop_vars('tile', errors='ignore')

# =============================================================================
# MAIN
# =============================================================================

def main():
    cluster = LocalCluster(n_workers=DASK_WORKERS, threads_per_worker=DASK_THREADS, memory_limit=DASK_MEMORY)
    client = Client(cluster)
    print(f"[System] Dask Client: {client}")
    
    if TEST_MODE:
        print("\n" + "="*60)
        print("  WARNING: TEST_MODE IS ENABLED. PROCESSING 1 TIMESTEP ONLY.")
        print("="*60 + "\n")
    
    try:
        for key, config in GROUPS.items():
            print(f"\n[Processing] Group: {key}")
            
            # 1. Load Tiles
            print("  [Load] Loading datasets...")
            datasets = []
            try:
                for name, path in TILES.items():
                    # Load minimal chunks for metadata
                    ds = xr.open_dataset(
                        path, 
                        group=config['nc_group'], 
                        chunks={'time': 1, 'x': 2048, 'y': 2048}
                    )
                    ds = fix_coordinates(ds)
                    ds = preproc_tile(ds) # Slicing happens here if TEST_MODE=True
                    datasets.append(ds)
            except Exception as e:
                print(f"  [Error] Failed to load tiles for {key}: {e}")
                continue

            # 2. Build Grid
            print("  [Grid] Building virtual canvas...")
            template = create_virtual_1km_grid(datasets)
            
            # 3. Merge
            merged_ds = merge_group(datasets, template, config)
            
            # 4. Write
            output_filename = f"icesat2_1km_seamless_{config['output_suffix']}.zarr"
            out_path = os.path.join(OUTPUT_DIR, output_filename)
            
            merged_ds.rio.write_crs(CRS_EPSG, inplace=True)
            merged_ds.attrs['source'] = f"ICESat-2 ATL15 {key} (Merged)"
            if TEST_MODE:
                 merged_ds.attrs['test_mode'] = "True: Single timestep only"
            
            print(f"  [Write] Saving to {out_path}...")
            if os.path.exists(out_path):
                import shutil
                shutil.rmtree(out_path)
            
            merged_ds.to_zarr(out_path, mode='w', consolidated=True)
            print(f"  [Success] {output_filename} created.")

    finally:
        client.close()
        cluster.close()

if __name__ == "__main__":
    main()
