import xarray as xr
import numpy as np
import os
from dask.distributed import Client
from pyproj import Transformer

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

if __name__ == "__main__":
    # 1. SETUP DASK CLIENT
    # 16GB limit (4x4GB) prevents the crash while enabling parallelism
    client = Client(n_workers=4, threads_per_worker=1, memory_limit='4GB')
    print(f">>> ⚡ Dask Client Running: {client.dashboard_link}")
    print(f">>> 🛰️ PROCESSING GRACE (Nearest Neighbor - Preserving Raw Values)...")

    # =========================================================
    # 2. PREPARE TARGET GRID
    # =========================================================
    print("    Preparing Target Grid...", end=" ")
    ds_target = xr.open_dataset(MASTER_GRID)

    # Calculate Lat/Lon for every 500m pixel (Query Points)
    X, Y = np.meshgrid(ds_target['x'].values, ds_target['y'].values)
    transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    lon_c, lat_c = transformer.transform(X, Y)

    ds_target['lat'] = (('y', 'x'), lat_c)
    ds_target['lon'] = (('y', 'x'), lon_c)
    ds_target = ds_target.set_coords(['lat', 'lon'])
    print("✅ Done.")

    # =========================================================
    # 3. PREPARE SOURCE DATA
    # =========================================================
    print("    Loading Source Data...", end=" ")

    # Chunking time=1 enables Dask Streaming
    ds_mass = xr.open_dataset(GRACE_MASS_FILE, chunks={'time': 1})[['lwe_thickness']]
    ds_mass = ds_mass.rename({'lwe_thickness': 'lwe'})

    ds_land = xr.open_dataset(LAND_MASK_FILE, chunks={})[['LO_val']].rename({'LO_val': 'mask_land'})
    ds_ocean = xr.open_dataset(OCEAN_MASK_FILE, chunks={})[['LO_val']].rename({'LO_val': 'mask_ocean'})

    ds_source = xr.merge([ds_mass, ds_land, ds_ocean])

    # Standardize names for interpolation
    rename_dict = {}
    if 'latitude' in ds_source.coords: rename_dict['latitude'] = 'lat'
    if 'longitude' in ds_source.coords: rename_dict['longitude'] = 'lon'
    ds_source = ds_source.rename(rename_dict)
    print("✅ Done.")

    # =========================================================
    # 4. INTERPOLATION (Nearest Neighbor)
    # =========================================================
    print("    Interpolating (Nearest Neighbor)...", end=" ")

    # method='nearest' preserves the exact "block" value of the satellite data.
    # It does not smooth edges. It does not crash RAM.
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

    # Clean encoding to prevent chunk conflicts
    for var in ds_final.variables:
        ds_final[var].encoding.pop('chunks', None)

    # Output chunks optimized for Zarr reading later
    ds_final = ds_final.chunk({'time': 1, 'y': 2048, 'x': 2048})

    # Execute via Client
    ds_final.to_zarr(OUTPUT_ZARR, mode='w', computed=True)

    print("✅ SAVED.")
    print(">>> 🏁 PHASE 1 (GRACE) COMPLETE.")
    
    client.close()
