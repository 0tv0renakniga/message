import xarray as xr
import numpy as np
import os
import shutil
from dask.distributed import Client
from pyproj import Transformer
import zarr

# --- CONFIGURATION ---
BATCH_SIZE = 1  # Process 1 year at a time to keep graph small
RAW_DIR = "data/raw/grace"
PROC_DIR = "processed_layers"

GRACE_MASS_FILE = f"{RAW_DIR}/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections_v02.nc"
LAND_MASK_FILE  = f"{RAW_DIR}/CSR_GRACE_GRACE-FO_RL06_Mascons_LandMask.nc"
OCEAN_MASK_FILE = f"{RAW_DIR}/CSR_GRACE_GRACE-FO_RL06_Mascons_OceanMask.nc"
MASTER_GRID     = f"{PROC_DIR}/master_grid_template.nc"
OUTPUT_ZARR     = f"{PROC_DIR}/grace_500m.zarr"

# Safety Check
# Cleanup previous runs
if os.path.exists(OUTPUT_ZARR):
    shutil.rmtree(OUTPUT_ZARR)

if __name__ == "__main__":
    # 1. SETUP CLIENT
    # Keep memory limit strict to detect issues early
    client = Client(n_workers=4, threads_per_worker=1, memory_limit='4GB')
    print(f">>> ⚡ Dask Client Running: {client.dashboard_link}")

    # =========================================================
    # 2. PREPARE TARGET GRID (LAZY MODE)
    # =========================================================
    print("    [1/5] Preparing Target Grid...", end=" ")
    try:
        ds_target = xr.open_dataset(MASTER_GRID, engine='h5netcdf')
    except:
        ds_target = xr.open_dataset(MASTER_GRID)

    # Calculate coordinates using Numpy (this is fast)
    X, Y = np.meshgrid(ds_target['x'].values, ds_target['y'].values)
    transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    lon_c, lat_c = transformer.transform(X, Y)

    ds_target['lat'] = (('y', 'x'), lat_c)
    ds_target['lon'] = (('y', 'x'), lon_c)
    ds_target = ds_target.set_coords(['lat', 'lon'])

    # 🟢 CRITICAL FIX: Chunk the target grid immediately.
    # This turns the lat/lon coordinates into Dask arrays. 
    # Any operation using these coords (like interp) will now happen lazily.
    ds_target = ds_target.chunk({'y': 2048, 'x': 2048})
    print("Done (Chunked).")

    # =========================================================
    # 3. PREPARE STATIC MASKS
    # =========================================================
    print("    [2/5] Processing Static Masks...", end=" ")

    # Load source masks (small enough for RAM)
    ds_land = xr.open_dataset(LAND_MASK_FILE, engine='h5netcdf')[['LO_val']].rename({'LO_val': 'mask_land'}).load()
    ds_ocean = xr.open_dataset(OCEAN_MASK_FILE, engine='h5netcdf')[['LO_val']].rename({'LO_val': 'mask_ocean'}).load()
    
    # Fix dims
    if 'latitude' in ds_land.coords: ds_land = ds_land.rename({'latitude': 'lat', 'longitude': 'lon'})
    if 'latitude' in ds_ocean.coords: ds_ocean = ds_ocean.rename({'latitude': 'lat', 'longitude': 'lon'})

    ds_masks = xr.merge([ds_land, ds_ocean])

    # 🟢 Convert source to Dask so interp is lazy
    ds_masks = ds_masks.chunk({'lat': -1, 'lon': -1}) 

    ds_masks_out = ds_masks.interp(
        lat=ds_target['lat'],
        lon=ds_target['lon'],
        method='nearest'
    )

    # Save Static Layers
    ds_static = xr.Dataset({
        'grace_land_frac': ds_masks_out['mask_land'],
        'grace_ocean_frac': ds_masks_out['mask_ocean']
    })
    
    # Write to Zarr
    ds_static.to_zarr(OUTPUT_ZARR, mode='w', compute=True)
    print("Done.")

    # =========================================================
    # 4. PREPARE SOURCE DATA
    # =========================================================
    print("    [3/5] Loading GRACE Metadata...", end=" ")
    ds_mass = xr.open_dataset(GRACE_MASS_FILE, engine='h5netcdf')[['lwe_thickness']]
    ds_mass = ds_mass.rename({'lwe_thickness': 'lwe'})
    if 'latitude' in ds_mass.coords: ds_mass = ds_mass.rename({'latitude': 'lat', 'longitude': 'lon'})
    print("Done.")

    # =========================================================
    # 5. BATCH PROCESSING LOOP (Streamlined)
    # =========================================================
    times = ds_mass.time.values
    total_steps = len(times)
    print(f"    [4/5] Batching {total_steps} time steps (Batch Size: {BATCH_SIZE})...")

    # Loop exactly like your reference script
    for i in range(0, total_steps, BATCH_SIZE):
        
        # 1. Select the batch
        current_times = times[i : i + BATCH_SIZE]
        
        # 🟢 CRITICAL FIX: Load data to RAM for speed, BUT immediately chunk it.
        # This tricks xarray into using Dask for the subsequent 'interp', 
        # preventing it from allocating the full 500m grid in RAM at once.
        ds_batch = ds_mass.sel(time=current_times).load()
        ds_batch = ds_batch.chunk({'time': 1}) 

        # 2. Interpolate (Lazy due to chunks)
        ds_batch_out = ds_batch.interp(
            lat=ds_target['lat'],
            lon=ds_target['lon'],
            method='nearest'
        )

        # 3. Re-chunk for Zarr structure
        ds_batch_out = ds_batch_out.chunk({'time': 1, 'y': 2048, 'x': 2048})
        
        # Clean encoding
        if 'chunks' in ds_batch_out['lwe'].encoding:
            del ds_batch_out['lwe'].encoding['chunks']

        # 4. Write
        if i == 0:
            # First batch: Append the NEW variable 'lwe' to existing store (from step 3)
            # mode='a' is correct here because the Zarr store exists (contains static masks)
            # but we are adding a new variable.
            ds_batch_out.to_zarr(OUTPUT_ZARR, mode='a', compute=True)
        else:
            # Subsequent batches: Append along Time
            ds_batch_out.to_zarr(OUTPUT_ZARR, mode='a', append_dim='time', compute=True)

        print(f"        Batch {i+1}-{min(i+BATCH_SIZE, total_steps)} / {total_steps} saved.")

    # =========================================================
    # 6. FINALIZE
    # =========================================================
    print("    [5/5] Consolidating Metadata...", end=" ")
    zarr.consolidate_metadata(OUTPUT_ZARR)
    print("Done.")
    
    client.close()
