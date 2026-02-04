import xarray as xr
import xesmf as xe
import numpy as np
import os
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

print(f">>> 🛰️ PROCESSING GRACE (Serial & Streamed)...")

# =========================================================
# 1. PREPARE TARGET GRID (Load into RAM for Regridder Build)
# =========================================================
print("    Preparing Target Grid...", end=" ")

# Open standard (Not chunked yet - we want coordinates in RAM)
ds_target = xr.open_dataset(MASTER_GRID)

# A. Generate Bounds (2D Corners, N+1 Shape)
rows = ds_target.sizes['y']
cols = ds_target.sizes['x']

# Use linspace for exact alignment
x_min = ds_target['x'].values[0] - 250.0
x_max = ds_target['x'].values[-1] + 250.0
y_max = ds_target['y'].values[0] + 250.0
y_min = ds_target['y'].values[-1] - 250.0

x_b = np.linspace(x_min, x_max, cols + 1)
y_b = np.linspace(y_max, y_min, rows + 1)
X_b, Y_b = np.meshgrid(x_b, y_b)
X, Y = np.meshgrid(ds_target['x'].values, ds_target['y'].values)

# B. Reproject
transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
lon_b, lat_b = transformer.transform(X_b, Y_b)
lon_c, lat_c = transformer.transform(X, Y)

# C. Assign Simple Names (lat/lon) and Set Coordinates
ds_target['lat'] = (('y', 'x'), lat_c)
ds_target['lon'] = (('y', 'x'), lon_c)
ds_target['lat_b'] = (('y_b', 'x_b'), lat_b)
ds_target['lon_b'] = (('y_b', 'x_b'), lon_b)

ds_target = ds_target.set_coords(['lat', 'lon', 'lat_b', 'lon_b'])
print("✅ Done.")

# =========================================================
# 2. PREPARE SOURCE GRID (Load into RAM for Regridder Build)
# =========================================================
print("    Preparing Source Grid...", end=" ")

# Load just the coordinates first
ds_mass_coords = xr.open_dataset(GRACE_MASS_FILE)[['lat', 'lon']]
ds_mass_coords = ds_mass_coords.rename({'lat': 'lat', 'lon': 'lon'}) # Ensure simple names

# Manual 1D Bounds (N+1)
lat_vals = ds_mass_coords.lat.values
lon_vals = ds_mass_coords.lon.values
lat_b = np.concatenate([lat_vals - 0.125, [lat_vals[-1] + 0.125]])
lon_b = np.concatenate([lon_vals - 0.125, [lon_vals[-1] + 0.125]])

ds_mass_coords = ds_mass_coords.assign_coords({'lat_b': lat_b, 'lon_b': lon_b})
print("✅ Done.")

# =========================================================
# 3. BUILD REGRIDDER (Serial - Calculates Weights Once)
# =========================================================
print("    Building Regridder (This takes ~5-10 mins)...")
# We pass the lightweight coordinate datasets, not the full data
# This computes the Sparse Matrix (~2GB RAM)
regridder = xe.Regridder(
    ds_mass_coords, 
    ds_target, 
    method='conservative_normed',
    periodic=True
)
print("✅ Regridder Built.")

# =========================================================
# 4. APPLY & STREAM (Lazy Loading)
# =========================================================
print("    Loading Data Lazily...", end=" ")

# NOW we load the actual heavy data with Dask
ds_mass = xr.open_dataset(GRACE_MASS_FILE, chunks={'time': 1})[['lwe_thickness']]
ds_mass = ds_mass.rename({'lwe_thickness': 'lwe'})
if 'latitude' in ds_mass.coords: ds_mass = ds_mass.rename({'latitude': 'lat', 'longitude': 'lon'})

# Masks (Static)
ds_land = xr.open_dataset(LAND_MASK_FILE, chunks={})[['LO_val']].rename({'LO_val': 'mask_land'})
ds_ocean = xr.open_dataset(OCEAN_MASK_FILE, chunks={})[['LO_val']].rename({'LO_val': 'mask_ocean'})
for ds in [ds_land, ds_ocean]:
    if 'latitude' in ds.coords: ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})

# Merge Source
ds_source = xr.merge([ds_mass, ds_land, ds_ocean])
print("✅ Done.")

print("    Regridding (Lazy Graph Construction)...", end=" ")
# xESMF allows applying a serial regridder to Dask arrays!
# It will broadcast the weight multiplication across the chunks.
lwe_500m = regridder(ds_source['lwe']).astype('float32')
land_frac = regridder(ds_source['mask_land']).astype('float32')
ocean_frac = regridder(ds_source['mask_ocean']).astype('float32')
print("✅ Done.")

# =========================================================
# 5. SAVE
# =========================================================
print(f"    Streaming to {OUTPUT_ZARR}...", end=" ")
ds_out = xr.Dataset({
    'lwe_thickness': lwe_500m,
    'grace_land_frac': land_frac,
    'grace_ocean_frac': ocean_frac
})

# Clear encoding
for var in ds_out.variables:
    ds_out[var].encoding.pop('chunks', None)

# Output chunks (Time=1 is critical for streaming)
ds_out = ds_out.chunk({'time': 1, 'y': 2048, 'x': 2048})

# Execution
ds_out.to_zarr(OUTPUT_ZARR, mode='w', computed=True)

print("✅ SAVED.")
print(">>> 🏁 PHASE 1 (GRACE) COMPLETE.")
