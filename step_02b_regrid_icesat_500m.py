"""
step_02b_regrid_icesat_500m.py

PRINCIPAL ENGINEER NOTES:
-------------------------
1. Objective: Upsample 1km seamless Zarrs to the 500m Master Grid.
2. Physics: Strictly Bilinear Interpolation (Source [4]).
3. Operations: Includes TEST_MODE for rapid validation.
"""

import os
import warnings
import xarray as xr
import xesmf as xe
import numpy as np
from dask.distributed import Client, LocalCluster

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

# >>> TEST FLAG <<<
TEST_MODE = True  # Set to False for Production Run

INPUT_DIR = "data/processed_layers/intermediate"
OUTPUT_DIR = "data/processed_layers"
MASTER_GRID = "data/processed_layers/master_grid_template.nc"

TASKS = {
    'deltah': 'icesat2_500m_deltah.zarr',
    'lag1': 'icesat2_500m_lag1.zarr',
    'lag4': 'icesat2_500m_lag4.zarr'
}

DASK_WORKERS = 8
DASK_MEMORY = "32GB"

# =============================================================================
# MAIN
# =============================================================================

def main():
    cluster = LocalCluster(n_workers=DASK_WORKERS, memory_limit=DASK_MEMORY)
    client = Client(cluster)
    print(f"[System] Dask Client: {client}")

    if TEST_MODE:
        print("\n" + "="*60)
        print("  WARNING: TEST_MODE IS ENABLED. PROCESSING 1 TIMESTEP ONLY.")
        print("="*60 + "\n")

    try:
        ds_target = xr.open_dataset(MASTER_GRID)
        
        for suffix, out_name in TASKS.items():
            in_name = f"icesat2_1km_seamless_{suffix}.zarr"
            in_path = os.path.join(INPUT_DIR, in_name)
            
            if not os.path.exists(in_path):
                print(f"[Skip] Input not found: {in_path}")
                continue
                
            print(f"\n[Processing] Regridding {in_name} -> {out_name}")
            
            ds_in = xr.open_zarr(in_path)
            
            # >>> TEST MODE SLICING <<<
            if TEST_MODE:
                # Even if input has 100 steps, we only take the first for regridding test
                ds_in = ds_in.isel(time=slice(0, 1))
            
            # Initialize Regridder (Bilinear)
            regridder = xe.Regridder(
                ds_in, 
                ds_target, 
                method='bilinear',
                filename='weights_icesat_1km_to_500m.nc',
                reuse_weights=True
            )
            
            # Apply Regridder
            ds_out = regridder(ds_in, keep_attrs=True)
            
            # Optimization & cleanup
            ds_out = ds_out.astype(np.float32)
            ds_out.rio.write_crs("EPSG:3031", inplace=True)
            
            if TEST_MODE:
                ds_out.attrs['test_mode'] = "True: Single timestep only"
            
            # Write
            out_path = os.path.join(OUTPUT_DIR, out_name)
            print(f"  [Write] Saving to {out_path}...")
            
            if os.path.exists(out_path):
                import shutil
                shutil.rmtree(out_path)
                
            ds_out.to_zarr(out_path, mode='w', consolidated=True)
            print("  [Success] Complete.")

    finally:
        client.close()
        cluster.close()

if __name__ == "__main__":
    main()
