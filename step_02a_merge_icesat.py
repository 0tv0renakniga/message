import xarray as xr
import numpy as np
import dask.array as da
from dask.distributed import Client, LocalCluster
from pathlib import Path
import logging
import shutil

# --- Configuration ---
TEST_MODE = True  # Set True to process only the first timestep
INPUT_DIR = Path("data/raw/icesat")
OUTPUT_DIR = Path("data/processed_layers")
INTERMEDIATE_ZARR = OUTPUT_DIR / "icesat2_1km_seamless.zarr"

# Corrected Configuration [1]
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

INPUT_CHUNKS = {'time': 1, 'y': 2000, 'x': 2000}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ICESat-Merge")

def get_dask_client():
    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=1,
        memory_limit='4GB',
        dashboard_address=':8787'
    )
    client = Client(cluster)
    logger.info(f"Dask Dashboard: {client.dashboard_link}")
    return client

def standardize_coordinates(ds: xr.Dataset) -> xr.Dataset:
    """Ensure coordinates are monotonic and named x/y for EPSG:3031 [2]."""
    if 'y' in ds.coords and ds.y < ds.y[-1]:
        ds = ds.sortby('y', ascending=False)  # North to South
    if 'x' in ds.coords and ds.x > ds.x[-1]:
        ds = ds.sortby('x', ascending=True)   # West to East
    return ds

def load_tile(filepath: Path, group: str) -> xr.Dataset:
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return None
    try:
        ds = xr.open_dataset(
            filepath, 
            group=group, 
            chunks=INPUT_CHUNKS, 
            engine='h5netcdf'
        )
        return standardize_coordinates(ds)
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        return None

def main():
    client = get_dask_client()
    
    if INTERMEDIATE_ZARR.exists() and TEST_MODE:
        shutil.rmtree(INTERMEDIATE_ZARR)

    try:
        # Iterate over the dictionary structure
        for group_name, config in GROUPS.items():
            logger.info(f"Processing group: {group_name} (Suffix: {config['suffix']})")
            
            datasets = []
            target_vars = config['vars']

            for quadrant in ['A1', 'A2', 'A3', 'A4']:
                # Pattern match for specific quadrant files
                files = list(INPUT_DIR.glob(f"ATL15_{quadrant}_*.nc"))
                if not files:
                    logger.warning(f"No file found for {quadrant}")
                    continue
                
                # Load tile and select specific variables
                ds = load_tile(files, group_name)
                if ds:
                    # Intersect requested vars with available vars
                    available = [v for v in target_vars if v in ds.variables]
                    datasets.append(ds[available])

            if not datasets:
                logger.error(f"No valid datasets for {group_name}")
                continue

            # --- Merge ---
            logger.info(f"  Merging {len(datasets)} tiles...")
            ds_merged = xr.combine_by_coords(
                datasets,
                coords='minimal',
                compat='override',
                combine_attrs='drop'
            )

            # --- Test Slice ---
            if TEST_MODE:
                logger.info("  TEST_MODE: Slicing time=0")
                ds_merged = ds_merged.isel(time=slice(0, 1))

            # --- Save ---
            # Use the group_name for the internal Zarr structure
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
        cluster.close()

if __name__ == "__main__":
    main()
