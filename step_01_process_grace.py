import xarray as xr
import xesmf as xe
import numpy as np
import os
import rioxarray
import cf_xarray
import dask.array as da  # <--- NEW IMPORT
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

print(f">>> 🛰️ PROCESSING GRACE (Parallel Mode with Dummy Mask)...")

# =========================================================
# 1. PREPARE TARGET GRID (With Dummy Mask)
# =========================================================
print("    Preparing Target Grid & Bounds...", end=" ")

# Open WITHOUT chunks first (since there are no vars to chunk yet)
ds_target = xr.open_dataset(MASTER_GRID)
ds_target.rio.write_crs("EPSG:3031", inplace=True)

# --- CRITICAL FIX: ADD DUMMY MASK FOR PARALLEL REGRIDDING ---
# This creates a variable full of 1s that is explicitly chunked.
# xESMF will use this to determine how to split the job.
rows = ds_target.sizes['y']
cols = ds_target.sizes['x']
chunks = (2048, 2048)

ds_target['mask'] = (('y', 'x'), da.ones((rows, cols), chunks=chunks, dtype='int8'))
# -----------------------------------------------------------

# Generate Manual Coordinates & Bounds
X, Y = np.meshgrid(ds_target['x'].values, ds_target['y'].values)

res = 500.0
x_b = np.arange(ds_target['x'][0] - res/2, ds_target['x'][-1] + res, res)
y_b = np.arange(ds_target['y'][0] + res/2, ds_target['y'][-1] - res, -res)
X_b, Y_b = np.meshgrid(x_b, y_b)

transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
lon_b, lat_b = transformer.transform(X_b, Y_b)
lon_c, lat_c = transformer.transform(X, Y)

ds_target['lat'] = (('y', 'x'), lat_c)
ds_target['lon'] = (('y', 'x'), lon_c)
ds_target['lat_b'] = (('y_b', 'x_b'), lat_b)
ds_target['lon_b'] = (('y_b', 'x_b'), lon_b)
print("✅ Done.")

# =========================================================
# 2. PREPARE SOURCE DATA (Lazy Loading)
# =========================================================
print("    Loading & Cleaning GRACE Data (Lazy)...", end=" ")

ds_mass = xr.open_dataset(GRACE_MASS_FILE, chunks={'time': 1})[['lwe_thickness']]
ds_mass = ds_mass.rename({'lwe_thickness': 'lwe'})

for coord in ['lat', 'lon']:
    if 'bounds' in ds_mass[coord].attrs:
        del ds_mass[coord].attrs['bounds']
ds_mass = ds_mass.cf.add_bounds(['lat', 'lon'])

ds_land = xr.open_dataset(LAND_MASK_FILE, chunks={})[['LO_val']].rename({'LO_val': 'mask_land'})
ds_ocean = xr.open_dataset(OCEAN_MASK_FILE, chunks={})[['LO_val']].rename({'LO_val': 'mask_ocean'})

for ds in [ds_land, ds_ocean]:
    for coord in ['lat', 'lon']:
        if 'bounds' in ds[coord].attrs:
            del ds[coord].attrs['bounds']

ds_source = xr.merge([ds_mass, ds_land, ds_ocean])
print("✅ Done.")

# =========================================================
# 3. REGRIDDING
# =========================================================
print("    Building Regridder (Parallel)...", end=" ")
regridder = xe.Regridder(
    ds_source, 
    ds_target, 
    method='conservative_normed',
    periodic=True,
    parallel=True 
)
print("✅ Built.")

print("    Regridding (Lazy)...", end=" ")
lwe_500m = regridder(ds_source['lwe']).astype('float32')
land_frac = regridder(ds_source['mask_land']).astype('float32')
ocean_frac = regridder(ds_source['mask_ocean']).astype('float32')
print("✅ Done.")

# =========================================================
# 4. SAVE
# =========================================================
print(f"    Streaming to {OUTPUT_ZARR}...", end=" ")
ds_out = xr.Dataset({
    'lwe_thickness': lwe_500m,
    'grace_land_frac': land_frac,
    'grace_ocean_frac': ocean_frac
})

# Re-chunk for writing
ds_out = ds_out.chunk({'time': 1, 'y': 2048, 'x': 2048})
ds_out.to_zarr(OUTPUT_ZARR, mode='w', computed=True)

print("✅ SAVED.")
print(">>> 🏁 PHASE 1 (GRACE) COMPLETE.")
