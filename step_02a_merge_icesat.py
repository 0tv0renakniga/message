import xarray as xr
import numpy as np
import dask.array as da
from dask.distributed import Client, LocalCluster
from pathlib import Path
import logging
import shutil

# --- Configuration ---
TEST_MODE = True  # Set to False for full run
INPUT_DIR = Path("data/raw/icesat")
OUTPUT_DIR = Path("data/processed_layers")
INTERMEDIATE_ZARR = OUTPUT_DIR / "icesat2_1km_seamless.zarr"

# Variable groups in ATL15 to process [3]
GROUPS = {
    'delta_h': ['delta_h', 'delta_h_sigma', 'ice_area'],
    'dhdt_lag1': ['dhdt', 'dhdt_sigma'],
    # Add lag4 or others if needed
}

# Chunking strategy for ~2GB files [2, 4]
# We keep chunks relatively small to allow Dask to swap them in/out
INPUT_CHUNKS = {'time': 1, 'y': 2000, 'x': 2000}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ICESat-Merge")

def get_dask_client():
    """Start a local Dask cluster with memory limits to prevent OOM."""
    cluster = LocalCluster(
        n_workers=4,           # Adjust based on CPU cores
        threads_per_worker=1,
        memory_limit='4GB',    # Conservative limit per worker
        dashboard_address=':8787'
    )
    client = Client(cluster)
    logger.info(f"Dask Dashboard available at: {client.dashboard_link}")
    return client

def load_tile(filepath: Path, group: str) -> xr.Dataset:
    """Load a single tile lazily with specific chunks."""
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return None
        
    try:
        # Open dataset with chunks to ensure lazy Dask arrays [5]
        ds = xr.open_dataset(filepath, group=group, chunks=INPUT_CHUNKS)
        
        # Standardize coordinates (ATL15 is typically y-descending)
        if 'y' in ds.coords:
            ds = ds.sortby('y', ascending=False)
        if 'x' in ds.coords:
            ds = ds.sortby('x', ascending=True)
            
        return ds
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        return None

def main():
    client = get_dask_client()
    
    # Clean up previous intermediate store if starting fresh
    if INTERMEDIATE_ZARR.exists() and TEST_MODE:
        shutil.rmtree(INTERMEDIATE_ZARR)

    try:
        for group_name, variables in GROUPS.items():
            logger.info(f"Processing group: {group_name}")
            
            datasets = []
            # Load A1, A2, A3, A4
            for quadrant in ['A1', 'A2', 'A3', 'A4']:
                # Find file matching pattern
                files = list(INPUT_DIR.glob(f"ATL15_{quadrant}_*.nc"))
                if not files:
                    logger.warning(f"No file found for {quadrant}")
                    continue
                
                ds = load_tile(files, group_name)
                if ds:
                    # Select only necessary variables to save memory
                    datasets.append(ds[variables])

            if not datasets:
                logger.error(f"No datasets loaded for {group_name}. Skipping.")
                continue

            # --- Merge Step ---
            logger.info("  Merging tiles...")
            # combine_by_coords handles the spatial assembly of A1-A4 [1]
            ds_merged = xr.combine_by_coords(
                datasets, 
                compat='override',  # Assume coordinates align on grid
                combine_attrs='drop',
                coords='minimal'
            )

            # --- Test Mode Slice ---
            if TEST_MODE:
                logger.info("  TEST_MODE: Slicing first timestep only.")
                ds_merged = ds_merged.isel(time=slice(0, 1))

            # --- Save to Intermediate Zarr ---
            # Save each group to a sub-path in the Zarr store
            group_path = INTERMEDIATE_ZARR / group_name
            
            logger.info(f"  Saving to {group_path}...")
            ds_merged.to_zarr(
                group_path, 
                mode='w', 
                consolidated=True,
                compute=True # Trigger Dask computation here
            )
            logger.info(f"  Finished {group_name}")

    finally:
        client.close()

if __name__ == "__main__":
    main()
