import xarray as xr
import numpy as np
import os
import dask.array as da
from dask.distributed import Client
from pyproj import Transformer
import shutil

# --- PATHS ---
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

# Cleanup
if os.path.exists(OUTPUT_ZARR):
    shutil.rmtree(OUTPUT_ZARR)

if __name__ == "__main__":
    # 1. SETUP CLIENT
    # 4 workers, 5GB limit each = 20GB total usage (safe zone)
    client = Client(n_workers=4, threads_per_worker=1, memory_limit='5GB')
    print(f">>> ⚡ Dask Client Running: {client.dashboard_link}")
    print(f">>> 🛰️ PROCESSING GRACE (Optimized Chunk Size)...")

    # =========================================================
    # 2. PREPARE TARGET GRID (Smaller Chunks)
    # =========================================================
    print("    Preparing Target Grid...", end=" ")
    try:
        ds_target = xr.open_dataset(MASTER_GRID, engine='h5netcdf')
    except:
        ds_target = xr.open_dataset(MASTER_GRID)

    X, Y = np.meshgrid(ds_target['x'].values, ds_target['y'].values)
    transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    lon_c, lat_c = transformer.transform(X, Y)

    # --- ADJUSTMENT: SMALLER CHUNKS (1024) ---
    # This keeps graph tasks small (~4MB) so the scheduler doesn't choke.
    CHUNK_SIZE = 1024
    lat_dask = da.from_array(lat_c, chunks=(CHUNK_SIZE, CHUNK_SIZE))
    lon_dask = da.from_array(lon_c, chunks=(CHUNK_SIZE, CHUNK_SIZE))

    ds_target['lat'] = (('y', 'x'), lat_dask)
    ds_target['lon'] = (('y', 'x'), lon_dask)
    ds_target = ds_target.set_coords(['lat', 'lon'])
    print("✅ Done.")

    # =========================================================
    # 3. PREPARE SOURCE DATA
    # =========================================================
    print("    Loading Source Data...", end=" ")
    
    # --- ADJUSTMENT: TIME BATCHING ---
    # Process 5 time steps at once. '1' is too overhead-heavy, 'All' is too big.
    ds_mass = xr.open_dataset(GRACE_MASS_FILE, chunks={'time': 5}, engine='h5netcdf')[['lwe_thickness']]
    ds_mass = ds_mass.rename({'lwe_thickness': 'lwe'})

    ds_land = xr.open_dataset(LAND_MASK_FILE, chunks={}, engine='h5netcdf')[['LO_val']].rename({'LO_val': 'mask_land'})
    ds_ocean = xr.open_dataset(OCEAN_MASK_FILE, chunks={}, engine='h5netcdf')[['LO_val']].rename({'LO_val': 'mask_ocean'})

    ds_source = xr.merge([ds_mass, ds_land, ds_ocean])

    # Standardize Names
    rename_dict = {}
    if 'latitude' in ds_source.coords: rename_dict['latitude'] = 'lat'
    if 'longitude' in ds_source.coords: rename_dict['longitude'] = 'lon'
    ds_source = ds_source.rename(rename_dict)
    
    print("✅ Done.")

    # =========================================================
    # 4. INTERPOLATION (Dask Distributed)
    # =========================================================
    print("    Interpolating (Nearest Neighbor)...", end=" ")

    # Because target lat/lon are Dask arrays (chunks=1024), 
    # this creates a clean, lightweight task graph.
    ds_out = ds_source.interp(
        lat=ds_target['lat'], 
        lon=ds_target['lon'], 
        method='nearest'
    )
    print("✅ Graph Built.")

    # =========================================================
    # 5. SAVE
    # =========================================================
    print(f"    Streaming to {OUTPUT_ZARR}...", end=" ")

    ds_final = xr.Dataset({
        'lwe_thickness': ds_out['lwe'],
        'grace_land_frac': ds_out['mask_land'],
        'grace_ocean_frac': ds_out['mask_ocean']
    })

    # Clear encoding to prevent conflicts
    for var in ds_final.variables:
        ds_final[var].encoding.pop('chunks', None)

    # Output chunks: 1 time step, but larger spatial blocks for faster reading
    ds_final = ds_final.chunk({'time': 1, 'y': 2048, 'x': 2048})

    # Run it
    ds_final.to_zarr(OUTPUT_ZARR, mode='w', computed=True)

    print("✅ SAVED.")
    print(">>> 🏁 PHASE 1 (GRACE) COMPLETE.")
    
    client.close()
