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
WEIGHTS_FILE    = f"{PROC_DIR}/grace_bilinear_weights.nc" # New filename for new method

# Safety Check
for f in [GRACE_MASS_FILE, LAND_MASK_FILE, OCEAN_MASK_FILE, MASTER_GRID]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"❌ Missing file: {f}")

# Cleanup: Delete old/partial weights to prevent "Corrupted File" errors
if os.path.exists(WEIGHTS_FILE):
    os.remove(WEIGHTS_FILE)

print(f">>> 🛰️ PROCESSING GRACE (Bilinear Method - Fast & Low RAM)...")

# =========================================================
# 1. PREPARE TARGET GRID (Standard Setup)
# =========================================================
print("    Preparing Target Grid...", end=" ")
ds_target = xr.open_dataset(MASTER_GRID)

# Generate Coordinates (Centers only - Bilinear doesn't need bounds!)
rows, cols = ds_target.sizes['y'], ds_target.sizes['x']
X, Y = np.meshgrid(ds_target['x'].values, ds_target['y'].values)

transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
lon_c, lat_c = transformer.transform(X, Y)

ds_target['lat'] = (('y', 'x'), lat_c)
ds_target['lon'] = (('y', 'x'), lon_c)
ds_target = ds_target.set_coords(['lat', 'lon'])
print("✅ Done.")

# =========================================================
# 2. PREPARE SOURCE GRID (Centers Only)
# =========================================================
print("    Preparing Source Grid...", end=" ")
ds_mass_coords = xr.open_dataset(GRACE_MASS_FILE)[['lat', 'lon']]
if 'latitude' in ds_mass_coords: ds_mass_coords = ds_mass_coords.rename({'latitude': 'lat', 'longitude': 'lon'})

# Note: We DO NOT need to calculate bounds for Bilinear!
print("✅ Done.")

# =========================================================
# 3. BUILD REGRIDDER (Bilinear)
# =========================================================
print(f"    Building Regridder (Bilinear)...")
# This should take SECONDS, not minutes.
regridder = xe.Regridder(
    ds_mass_coords, 
    ds_target, 
    method='bilinear',   # <--- THE FIX
    periodic=True,
    filename=WEIGHTS_FILE
)
print("✅ Regridder Ready.")

# =========================================================
# 4. STREAMING PROCESSING
# =========================================================
print("    Processing Data Stream...", end=" ")

# Load Data Chunked
ds_mass = xr.open_dataset(GRACE_MASS_FILE, chunks={'time': 1})[['lwe_thickness']].rename({'lwe_thickness': 'lwe'})
if 'latitude' in ds_mass.coords: ds_mass = ds_mass.rename({'latitude': 'lat', 'longitude': 'lon'})

# Masks
ds_land = xr.open_dataset(LAND_MASK_FILE, chunks={})[['LO_val']].rename({'LO_val': 'mask_land'})
ds_ocean = xr.open_dataset(OCEAN_MASK_FILE, chunks={})[['LO_val']].rename({'LO_val': 'mask_ocean'})
for ds in [ds_land, ds_ocean]:
    if 'latitude' in ds.coords: ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})

ds_source = xr.merge([ds_mass, ds_land, ds_ocean])

# Apply Regridder
# Bilinear interpolation is just a weighted average of the 4 nearest neighbors. Very fast.
lwe_500m = regridder(ds_source['lwe']).astype('float32')
land_frac = regridder(ds_source['mask_land']).astype('float32')
ocean_frac = regridder(ds_source['mask_ocean']).astype('float32')
print("✅ Done.")

# =========================================================
# 5. SAVE
# =========================================================
print(f"    Writing to {OUTPUT_ZARR}...", end=" ")
ds_out = xr.Dataset({
    'lwe_thickness': lwe_500m,
    'grace_land_frac': land_frac,
    'grace_ocean_frac': ocean_frac
})

for var in ds_out.variables:
    ds_out[var].encoding.pop('chunks', None)

ds_out = ds_out.chunk({'time': 1, 'y': 2048, 'x': 2048})
ds_out.to_zarr(OUTPUT_ZARR, mode='w', computed=True)

print("✅ SAVED.")
print(">>> 🏁 PHASE 1 (GRACE) COMPLETE.")
