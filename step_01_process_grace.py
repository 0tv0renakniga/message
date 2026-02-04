import xarray as xr
import numpy as np
import os
import shutil
from dask.distributed import Client
from pyproj import Transformer
import zarr

# --- CONFIGURATION ---
BATCH_SIZE = 12  # Process 1 year at a time to keep graph small
RAW_DIR = "data/raw/grace"
PROC_DIR = "processed_layers"

GRACE_MASS_FILE = f"{RAW_DIR}/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections_v02.nc"
LAND_MASK_FILE  = f"{RAW_DIR}/CSR_GRACE_GRACE-FO_RL06_Mascons_LandMask.nc"
OCEAN_MASK_FILE = f"{RAW_DIR}/CSR_GRACE_GRACE-FO_RL06_Mascons_OceanMask.nc"
MASTER_GRID     = f"{PROC_DIR}/master_grid_template.nc"
OUTPUT_ZARR     = f"{PROC_DIR}/grace_500m.zarr"

# Safety Check
for f in [GRACE_MASS_FILE, LAND_MASK_FILE, OCEAN_MASK_FILE, MASTER_GRID]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"❌ Missing file: {f}")

# Cleanup previous runs
if os.path.exists(OUTPUT_ZARR):
    shutil.rmtree(OUTPUT_ZARR)

if __name__ == "__main__":
    # 1. SETUP CLIENT (As requested)
    # 4 workers, strict memory limit to force spilling if needed
    client = Client(n_workers=4, threads_per_worker=1, memory_limit='4GB')
    print(f">>> ⚡ Dask Client Running: {client.dashboard_link}")
    print(f">>> 🛰️ PROCESSING GRACE (Client + Batching)...")

    # =========================================================
    # 2. PREPARE TARGET GRID (Load to RAM)
    # =========================================================
    print("    [1/5] Loading Target Grid...", end=" ")
    try:
        ds_target = xr.open_dataset(MASTER_GRID, engine='h5netcdf')
    except:
        ds_target = xr.open_dataset(MASTER_GRID)

    # We use numpy for the coordinates to ensure 'interp' runs fast
    X, Y = np.meshgrid(ds_target['x'].values, ds_target['y'].values)
    transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    lon_c, lat_c = transformer.transform(X, Y)

    ds_target['lat'] = (('y', 'x'), lat_c)
    ds_target['lon'] = (('y', 'x'), lon_c)
    ds_target = ds_target.set_coords(['lat', 'lon'])
    print("Done.")

    # =========================================================
    # 3. PREPARE STATIC MASKS (One-off Interpolation)
    # =========================================================
    print("    [2/5] Processing Static Masks...", end=" ")
    
    ds_land = xr.open_dataset(LAND_MASK_FILE, engine='h5netcdf')[['LO_val']].load()
    ds_ocean = xr.open_dataset(OCEAN_MASK_FILE, engine='h5netcdf')[['LO_val']].load()

    # Manual Rename
    ds_land = ds_land.rename({'LO_val': 'mask_land'})
    if 'latitude' in ds_land.coords: ds_land = ds_land.rename({'latitude': 'lat', 'longitude': 'lon'})

    ds_ocean = ds_ocean.rename({'LO_val': 'mask_ocean'})
    if 'latitude' in ds_ocean.coords: ds_ocean = ds_ocean.rename({'latitude': 'lat', 'longitude': 'lon'})

    # Merge & Interpolate
    ds_masks = xr.merge([ds_land, ds_ocean])
    ds_masks_out = ds_masks.interp(
        lat=ds_target['lat'], 
        lon=ds_target['lon'], 
        method='nearest'
    )

    # Save Masks to Zarr (Creating the store)
    ds_static = xr.Dataset({
        'grace_land_frac': ds_masks_out['mask_land'],
        'grace_ocean_frac': ds_masks_out['mask_ocean']
    })
    
    # Chunking: Spatial only
    ds_static = ds_static.chunk({'y': 2048, 'x': 2048})
    ds_static.to_zarr(OUTPUT_ZARR, mode='w', computed=True)
    print("Done.")

    # =========================================================
    # 4. PREPARE SOURCE DATA
    # =========================================================
    print("    [3/5] Loading GRACE Data...", end=" ")
    # Load metadata only (not data)
    ds_mass = xr.open_dataset(GRACE_MASS_FILE, engine='h5netcdf')[['lwe_thickness']]
    ds_mass = ds_mass.rename({'lwe_thickness': 'lwe'})
    if 'latitude' in ds_mass.coords: ds_mass = ds_mass.rename({'latitude': 'lat', 'longitude': 'lon'})
    print("Done.")

    # =========================================================
    # 5. BATCH PROCESSING LOOP
    # =========================================================
    times = ds_mass.time.values
    total_steps = len(times)
    print(f"    [4/5] Batching {total_steps} time steps (Batch Size: {BATCH_SIZE})...")

    for i in range(0, total_steps, BATCH_SIZE):
        # Select Batch
        current_times = times[i : i + BATCH_SIZE]
        ds_batch = ds_mass.sel(time=current_times).load() # Load small batch to RAM to speed up interp

        # Interpolate Batch
        # Since ds_batch is in RAM, this is fast and uses the Client implicitly if chunks exist
        # (But here we loaded it to avoid graph overhead completely for the interp step)
        ds_batch_out = ds_batch.interp(
            lat=ds_target['lat'], 
            lon=ds_target['lon'], 
            method='nearest'
        )

        # Chunk for Zarr (1, 2048, 2048)
        ds_batch_out = ds_batch_out.chunk({'time': 1, 'y': 2048, 'x': 2048})

        # Remove encoding to prevent conflicts
        if 'chunks' in ds_batch_out['lwe'].encoding:
            del ds_batch_out['lwe'].encoding['chunks']

        # Append to Zarr
        # On first batch (i=0), we append the NEW variable 'lwe' to the existing store
        # On later batches, we append along the 'time' dimension
        if i == 0:
            ds_batch_out.to_zarr(OUTPUT_ZARR, mode='w', compute=True)
        else:
            ds_batch_out.to_zarr(OUTPUT_ZARR, mode='a', append_dim='time', computedTrue)

        print(f"       Batch {i}-{min(i+BATCH_SIZE, total_steps)} / {total_steps} saved.")

    # =========================================================
    # 6. FINALIZE
    # =========================================================
    print("    [5/5] Consolidating Metadata...", end=" ")
    zarr.consolidate_metadata(OUTPUT_ZARR)
    print("Done.")
    
    print(">>> 🏁 PHASE 1 (GRACE) COMPLETE.")
    client.close()
