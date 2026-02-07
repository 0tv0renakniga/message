"""
step_02a_merge_icesat_tiles.py

Phase: 1 (ICESat-2 Processing)
Goal: Merge 4 ATL15 tiles (A1-A4) into a seamless 1km Zarr dataset.
Critical Constraints:
    - Must handle ATL15 Group structure (/delta_h, /dhdt_lag1).
    - Must enforce coordinate monotonicity (y-descending, x-ascending).
    - Must decode custom time units (days since 2018-01-01).

References:
    - Master Plan Sec 3.2: Merging Strategy [5]
    - Plot Raw ICESat: Coordinate Monotonicity [1]
    - Inspect NetCDF: File Patterns [3]
"""

import xarray as xr
import numpy as np
import pandas as pd
import dask.config
from dask.distributed import Client, LocalCluster
from pathlib import Path
import logging
import shutil

# --- Configuration ---
TEST_MODE = True  # Set False for full production run
INPUT_DIR = Path("data/raw/icesat")
OUTPUT_DIR = Path("data/processed_layers")
INTERMEDIATE_ZARR = OUTPUT_DIR / "icesat2_1km_seamless.zarr"

# Corrected Configuration [3]
GROUPS = {
    'delta_h': {
        'vars': ['delta_h', 'delta_h_sigma', 'ice_area', 'data_count', 'misfit_rms'],
        'suffix': 'deltah'
    },
    'dhdt_lag1': {
        'vars': ['dhdt', 'dhdt_sigma', 'ice_area'],
        'suffix': 'lag1'
    },
    'dhdt_lag4': {
        'vars': ['dhdt', 'dhdt_sigma', 'ice_area'],
        'suffix': 'lag4'
    }
}

# Chunking for 1km input (approx 2GB per tile)
INPUT_CHUNKS = {'time': 1, 'y': 2000, 'x': 2000}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ICESat-Merge")

def get_dask_client():
    """Start LocalCluster. Note: HDF5 often needs single-threaded readers."""
    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=1,
        memory_limit='6GB', # 2GB tile + overhead
        dashboard_address=':8787'
    )
    client = Client(cluster)
    logger.info(f"Dask Dashboard: {client.dashboard_link}")
    return client

def decode_atl15_time(ds: xr.Dataset) -> xr.Dataset:
    """
    Decodes ATL15 time: 'days since 2018-01-01'.
    Standard xarray.open_dataset often fails on HDF5 groups without this.
    Reference: [4]
    """
    if 'time' in ds.coords:
        # Check if already decoded (datetime64)
        if not np.issubdtype(ds.time.dtype, np.datetime64):
            logger.info("  Decoding custom ATL15 time units...")
            # Create baseline
            base_date = pd.Timestamp("2018-01-01")
            # Convert values to timedeltas
            try:
                # Ensure we are working with the data, not dask array for small coords
                time_offsets = pd.to_timedelta(ds.time.values, unit='D')
                ds = ds.assign_coords(time=base_date + time_offsets)
            except Exception as e:
                logger.warning(f"  Time decoding warning: {e}")
    return ds

def standardize_coordinates(ds: xr.Dataset) -> xr.Dataset:
    """
    Enforces EPSG:3031 strict monotonicity.
    y must be descending (North -> South).
    x must be ascending (West -> East).
    Reference: [1] [2]
    """
    # 1. Rename dims if necessary (ATL15 usually correct, but safety first)
    # 2. Sort Y Descending
    if ds.y < ds.y[-1]:
        logger.info("  Sorting Y coordinate (ascending -> descending)")
        ds = ds.sortby('y', ascending=False)
    
    # 3. Sort X Ascending
    if ds.x > ds.x[-1]:
        logger.info("  Sorting X coordinate (descending -> ascending)")
        ds = ds.sortby('x', ascending=True)
        
    return ds

def load_tile(filepath: Path, group: str, vars_to_keep: list) -> xr.Dataset:
    """Load a single ATL15 tile group."""
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return None
    
    try:
        # Load specific group
        ds = xr.open_dataset(
            filepath, 
            group=group, 
            chunks=INPUT_CHUNKS,
            engine='h5netcdf' 
        )
        
        # Filter variables immediately to save memory
        # Only keep what exists in this tile
        valid_vars = [v for v in vars_to_keep if v in ds.variables]
        ds = ds[valid_vars]
        
        # Fix Time and Space
        ds = decode_atl15_time(ds)
        ds = standardize_coordinates(ds)
        
        return ds
    except Exception as e:
        logger.error(f"Failed to load {filepath} [{group}]: {e}")
        return None

def main():
    # HDF5 is not thread-safe by default, lock it just in case
    dask.config.set(scheduler='synchronized') 
    # Use standard client for computation
    client = get_dask_client()
    
    # Cleanup previous run
    if INTERMEDIATE_ZARR.exists() and TEST_MODE:
        shutil.rmtree(INTERMEDIATE_ZARR)

    try:
        for group_name, config in GROUPS.items():
            logger.info(f"Processing Group: {group_name} (Suffix: {config['suffix']})")
            
            datasets = []
            target_vars = config['vars']

            # Load A1-A4
            for quadrant in ['A1', 'A2', 'A3', 'A4']:
                # Flexible pattern match for version numbers (e.g., 0311, 0328)
                files = list(INPUT_DIR.glob(f"ATL15_{quadrant}_*.nc"))
                if not files:
                    logger.warning(f"  Missing tile: {quadrant}")
                    continue
                
                # Take the first match (assuming one version present)
                ds = load_tile(files, group_name, target_vars)
                if ds:
                    datasets.append(ds)

            if not datasets:
                logger.error(f"No datasets found for {group_name}. Skipping.")
                continue

            # --- Merge Step ---
            # Reference [5]: "If tiles DO overlap: Create mosaic...". 
            # combine_by_coords handles non-overlapping perfectly. 
            # For overlapping edges, we use 'compat="override"' to prioritize A1>A2>A3>A4 
            # effectively, unless we implement the heavy "sigma-min" reduction.
            # Given memory constraints, we use combine_by_coords for V1.
            logger.info(f"  Merging {len(datasets)} tiles...")
            
            try:
                ds_merged = xr.combine_by_coords(
                    datasets,
                    coords='minimal', 
                    compat='override', # Resolves overlaps by taking the first one
                    combine_attrs='drop'
                )
            except ValueError as e:
                logger.error(f"  Merge failed. Check coordinate alignment. Error: {e}")
                continue

            # --- Test Mode ---
            if TEST_MODE:
                logger.info("  TEST_MODE: Slicing first timestep.")
                ds_merged = ds_merged.isel(time=slice(0, 1))

            # --- Validation Fragment ---
            # Check for empty grid (common issue if coords misaligned)
            if ds_merged.nbytes == 0:
                logger.error("  Resulting dataset is empty! Check inputs.")
                continue

            # --- Save ---
            group_path = INTERMEDIATE_ZARR / group_name
            logger.info(f"  Saving to {group_path}...")
            
            ds_merged.to_zarr(
                group_path, 
                mode='w', 
                consolidated=True
            )
            logger.info(f"  Finished {group_name}")

    finally:
        client.close()

if __name__ == "__main__":
    main()
