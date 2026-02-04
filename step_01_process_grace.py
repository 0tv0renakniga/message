import xarray as xr
import xesmf as xe
import numpy as np
import os
import rioxarray
import cf_xarray
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

print(f">>> 🛰️ PROCESSING GRACE (Dask Parallel Mode)...")

# =========================================================
# 1. PREPARE TARGET GRID (Chunked + Manual Bounds)
# =========================================================
print("    Preparing Target Grid & Bounds...", end=" ")

# FIX: Chunk the target grid so Dask can write to it in parallel
ds_target = xr.open_dataset(MASTER_GRID, chunks={'y': 2048, 'x': 2048})
ds_target.rio.write_crs("EPSG:3031", inplace=True)

# Generate Coordinates (we use compute() here because coordinates must be in memory for the Transformer)
X, Y = np.meshgrid(ds_target['x'].values, ds_target['y'].values)

# Generate Manual Bounds for Conservation
res = 500.0
x_b = np.arange(ds_target['x'][0] - res/2, ds_target['x'][-1] + res, res)
y_b = np.arange(ds_target['y'][0] + res/2, ds_target['y'][-1] - res, -res)
X_b, Y_b = np.meshgrid(x_b, y_b)

# Transform to Lat/Lon
transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
lon_b, lat_b = transformer.transform(X_b, Y_b)
lon_c, lat_c = transformer.transform(X, Y)

# Assign back to dataset
ds_target['lat'] = (('y', 'x'), lat_c)
ds_target['lon'] = (('y', 'x'), lon_c)
ds_target['lat_b'] = (('y_b', 'x_b'), lat_b)
ds_target['lon_b'] = (('y_b', 'x_b'), lon_b)
print("✅ Done.")

# =========================================================
# 2. PREPARE SOURCE DATA (Lazy Loading)
# =========================================================
print("    Loading & Cleaning GRACE Data (Lazy)...", end=" ")

# CRITICAL: chunks={'time': 1} enables Dask Streaming
ds_mass = xr.open_dataset(GRACE_MASS_FILE, chunks={'time': 1})[['lwe_thickness']]
ds_mass = ds_mass.rename({'lwe_thickness': 'lwe'})

# Remove broken bounds if present
for coord in ['lat', 'lon']:
    if 'bounds' in ds_mass[coord].attrs:
        del ds_mass[coord].attrs['bounds']
# Add fresh bounds
ds_mass = ds_mass.cf.add_bounds(['lat', 'lon'])

# Masks (Static, but we chunk them to match the pipeline)
ds_land = xr.open_dataset(LAND_MASK_FILE, chunks={})[['LO_val']].rename({'LO_val': 'mask_land'})
ds_ocean = xr.open_dataset(OCEAN_MASK_FILE, chunks={})[['LO_val']].rename({'LO_val': 'mask_ocean'})

# Remove broken bounds from masks
for ds in [ds_land, ds_ocean]:
    for coord in ['lat', 'lon']:
        if 'bounds' in ds[coord].attrs:
            del ds[coord].attrs['bounds']

# Merge
ds_source = xr.merge([ds_mass, ds_land, ds_ocean])
print("✅ Done.")

# =========================================================
# 3. REGRIDDING
# =========================================================
print("    Building Regridder (Parallel)...", end=" ")
# parallel=True now works because both ds_source and ds_target are chunked
regridder = xe.Regridder(
    ds_source, 
    ds_target, 
    method='conservative_normed',
    periodic=True,
    parallel=True 
)
print("✅ Built.")

print("    Regridding (Lazy)...", end=" ")
# This builds the graph, does not compute yet
lwe_500m = regridder(ds_source['lwe']).astype('float32')
land_frac = regridder(ds_source['mask_land']).astype('float32')
ocean_frac = regridder(ds_source['mask_ocean']).astype('float32')
print("✅ Done.")

# =========================================================
# 4. SAVE (Stream to Disk)
# =========================================================
print(f"    Streaming to {OUTPUT_ZARR}...", end=" ")
ds_out = xr.Dataset({
    'lwe_thickness': lwe_500m,
    'grace_land_frac': land_frac,
    'grace_ocean_frac': ocean_frac
})

# Re-chunk for optimal Zarr writing
ds_out = ds_out.chunk({'time': 1, 'y': 2048, 'x': 2048})

# Write (The computation happens here)
ds_out.to_zarr(OUTPUT_ZARR, mode='w', computed=True)

print("✅ SAVED.")
print(">>> 🏁 PHASE 1 (GRACE) COMPLETE.")
