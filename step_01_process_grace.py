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

  # ... (Keep your existing Setup, Target Grid, and Static Mask sections) ...

    # =========================================================
    # 4. PREPARE SOURCE DATA
    # =========================================================
    print("    [3/5] Loading GRACE Metadata...", end=" ")
    ds_mass = xr.open_dataset(GRACE_MASS_FILE, engine='h5netcdf')[['lwe_thickness']]
    ds_mass = ds_mass.rename({'lwe_thickness': 'lwe'})
    if 'latitude' in ds_mass.coords: ds_mass = ds_mass.rename({'latitude': 'lat', 'longitude': 'lon'})
    print("Done.")

    # =========================================================
    # 5. PRE-ALLOCATE ZARR STORE (The Fix)
    # =========================================================
    print("    [4/5] Pre-allocating Zarr Store...", end=" ")
    
    # 1. Define the full shape of the output
    total_steps = len(ds_mass.time)
    ny, nx = ds_target['lat'].shape
    
    # 2. Create a "dummy" Dask array representing the full final variable
    #    We use the exact chunk size we want on disk (1, 2048, 2048)
    import dask.array as da
    dummy_data = da.empty((total_steps, ny, nx), chunks=(1, 2048, 2048), dtype=np.float32)
    
    # 3. Create a template dataset
    ds_template = xr.Dataset(
        {'lwe': (('time', 'y', 'x'), dummy_data)},
        coords={
            'time': ds_mass.time.values,
            'y': ds_target.y.values,
            'x': ds_target.x.values
        }
    )
    
    # 4. Write the METADATA only (compute=False)
    #    mode='a' appends this new variable to the store we created in step 3 (masks)
    ds_template.to_zarr(OUTPUT_ZARR, mode='a', compute=False)
    print(f"Done. (Created slots for {total_steps} time steps)")

    # =========================================================
    # 6. BATCH PROCESSING LOOP (Region Mode)
    # =========================================================
    print(f"    [5/5] Processing Batches (Batch Size: {BATCH_SIZE})...")

    for i in range(0, total_steps, BATCH_SIZE):
        # 1. Select Batch
        current_times = times[i : i + BATCH_SIZE]
        
        # 2. Load Source -> Chunk -> Interp
        #    We load source to RAM (small) then immediately chunk to force Lazy execution
        ds_batch = ds_mass.sel(time=current_times).load().chunk({'time': 1})
        
        ds_batch_out = ds_batch.interp(
            lat=ds_target['lat'],
            lon=ds_target['lon'],
            method='nearest'
        )

        # 3. Re-chunk to match the Zarr target exactly
        ds_batch_out = ds_batch_out.chunk({'time': 1, 'y': 2048, 'x': 2048})

        # 4. Write to Region
        #    We drop coordinates here because they are already written in the file.
        #    This prevents Xarray from trying to check/rewrite them, saving memory.
        ds_write = ds_batch_out[['lwe']].drop_vars(['time', 'y', 'x'])
        
        ds_write.to_zarr(
            OUTPUT_ZARR, 
            region={'time': slice(i, i + BATCH_SIZE)}
        )
        
        print(f"        Batch {i+1}-{min(i+BATCH_SIZE, total_steps)} written.")

    print(">>> 🏁 PHASE 1 (GRACE) COMPLETE.")
    client.close() 
