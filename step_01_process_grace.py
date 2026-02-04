import xarray as xr
import xesmf as xe
import numpy as np
import os
import rioxarray
import cf_xarray
import dask.array as da
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

print(f">>> 🛰️ PROCESSING GRACE (Final Alignment Fix)...")

# =========================================================
# 1. PREPARE TARGET GRID (Strict Bounds Math)
# =========================================================
print("    Preparing Target Grid...", end=" ")

ds_target = xr.open_dataset(MASTER_GRID)
ds_target.rio.write_crs("EPSG:3031", inplace=True)

# A. Add Dummy Mask for Parallel Chunks
rows = ds_target.sizes['y']
cols = ds_target.sizes['x']
chunks = (2048, 2048)
ds_target['mask'] = (('y', 'x'), da.ones((rows, cols), chunks=chunks, dtype='int8'))

# B. Generate Coordinates (2D)
X, Y = np.meshgrid(ds_target['x'].values, ds_target['y'].values)

# C. Generate Bounds (Using Linspace to guarantee N+1)
# Grid definition: -3,072,000 to 3,072,000.
# Left Edge: x[0] - 250. Right Edge: x[-1] + 250.
x_min = ds_target['x'].values[0] - 250.0
x_max = ds_target['x'].values[-1] + 250.0
y_max = ds_target['y'].values[0] + 250.0  # Top (Y is inverted)
y_min = ds_target['y'].values[-1] - 250.0

# Force exact count: Pixels + 1
x_b = np.linspace(x_min, x_max, cols + 1)
y_b = np.linspace(y_max, y_min, rows + 1) # Note y_max first for descending Y
X_b, Y_b = np.meshgrid(x_b, y_b)

# D. Reproject to Lat/Lon
transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
lon_b, lat_b = transformer.transform(X_b, Y_b)
lon_c, lat_c = transformer.transform(X, Y)

# E. Assign with Standard Names
ds_target['latitude'] = (('y', 'x'), lat_c)
ds_target['longitude'] = (('y', 'x'), lon_c)
ds_target['lat_b'] = (('y_b', 'x_b'), lat_b)
ds_target['lon_b'] = (('y_b', 'x_b'), lon_b)

# F. Attributes (Critical for cf-xarray)
ds_target['latitude'].attrs = {'units': 'degrees_north', 'standard_name': 'latitude'}
ds_target['longitude'].attrs = {'units': 'degrees_east', 'standard_name': 'longitude'}
ds_target['lat_b'].attrs = {'units': 'degrees_north'}
ds_target['lon_b'].attrs = {'units': 'degrees_east'}

print("✅ Done.")

# =========================================================
# 2. PREPARE SOURCE DATA (Standard Names)
# =========================================================
print("    Loading Source Data...", end=" ")

# Load Mass
ds_mass = xr.open_dataset(GRACE_MASS_FILE, chunks={'time': 1})[['lwe_thickness']]
ds_mass = ds_mass.rename({'lwe_thickness': 'lwe'})

# Rename lat/lon -> latitude/longitude
if 'lat' in ds_mass: ds_mass = ds_mass.rename({'lat': 'latitude', 'lon': 'longitude'})

# Clean Bounds
for coord in ['latitude', 'longitude']:
    if 'bounds' in ds_mass[coord].attrs: del ds_mass[coord].attrs['bounds']
ds_mass = ds_mass.cf.add_bounds(['latitude', 'longitude'])

# Load Masks
ds_land = xr.open_dataset(LAND_MASK_FILE, chunks={})[['LO_val']].rename({'LO_val': 'mask_land'})
ds_ocean = xr.open_dataset(OCEAN_MASK_FILE, chunks={})[['LO_val']].rename({'LO_val': 'mask_ocean'})

# Rename and Clean Masks
for ds in [ds_land, ds_ocean]:
    if 'lat' in ds: ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})
    for coord in ['latitude', 'longitude']:
        if 'bounds' in ds[coord].attrs: del ds[coord].attrs['bounds']

# Merge
ds_source = xr.merge([ds_mass, ds_land, ds_ocean])
print("✅ Done.")

# =========================================================
# 3. REGRIDDING
# =========================================================
print("    Building Regridder...", end=" ")
regridder = xe.Regridder(
    ds_source, 
    ds_target, 
    method='conservative_normed',
    periodic=True,
    parallel=True 
)
print("✅ Built.")

print("    Regridding...", end=" ")
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

# Explicitly clear encoding to stop the "chunks separate" warning
for var in ds_out.variables:
    ds_out[var].encoding.pop('chunks', None)

ds_out = ds_out.chunk({'time': 1, 'y': 2048, 'x': 2048})
ds_out.to_zarr(OUTPUT_ZARR, mode='w', computed=True)

print("✅ SAVED.")
print(">>> 🏁 PHASE 1 (GRACE) COMPLETE.")
