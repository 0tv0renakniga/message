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

print(f">>> 🛰️ PROCESSING GRACE (Serial Mode - RAM Safe)...")

# =========================================================
# 1. PREPARE TARGET GRID
# =========================================================
print("    Preparing Target Grid...", end=" ")

ds_target = xr.open_dataset(MASTER_GRID)

# Generate Coordinates (2D Centers)
X, Y = np.meshgrid(ds_target['x'].values, ds_target['y'].values)

# Generate Bounds (2D Corners, N+1 Shape)
rows = ds_target.sizes['y']
cols = ds_target.sizes['x']
x_min = ds_target['x'].values[0] - 250.0
x_max = ds_target['x'].values[-1] + 250.0
y_max = ds_target['y'].values[0] + 250.0
y_min = ds_target['y'].values[-1] - 250.0

x_b = np.linspace(x_min, x_max, cols + 1)
y_b = np.linspace(y_max, y_min, rows + 1)
X_b, Y_b = np.meshgrid(x_b, y_b)

# Reproject to Lat/Lon
transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
lon_b, lat_b = transformer.transform(X_b, Y_b) # Corners
lon_c, lat_c = transformer.transform(X, Y)     # Centers

# Assign Variables
ds_target['lat'] = (('y', 'x'), lat_c)
ds_target['lon'] = (('y', 'x'), lon_c)
ds_target['lat_b'] = (('y_b', 'x_b'), lat_b)
ds_target['lon_b'] = (('y_b', 'x_b'), lon_b)

# Attributes (Only on centers to please xESMF/CF)
ds_target['lat'].attrs = {'units': 'degrees_north', 'standard_name': 'latitude'}
ds_target['lon'].attrs = {'units': 'degrees_east', 'standard_name': 'longitude'}

print("✅ Done.")

# =========================================================
# 2. PREPARE SOURCE DATA
# =========================================================
print("    Loading Source Data...", end=" ")

# Load Mass - NO CHUNKS YET (Load small source grid into RAM)
ds_mass = xr.open_dataset(GRACE_MASS_FILE)[['lwe_thickness']]
ds_mass = ds_mass.rename({'lwe_thickness': 'lwe'})

rename_dict = {}
if 'latitude' in ds_mass: rename_dict['latitude'] = 'lat'
if 'longitude' in ds_mass: rename_dict['longitude'] = 'lon'
ds_mass = ds_mass.rename(rename_dict)

# Manual Bounds (N+1)
lat_vals = ds_mass.lat.values
lon_vals = ds_mass.lon.values
lat_b = np.concatenate([lat_vals - 0.125, [lat_vals[-1] + 0.125]])
lon_b = np.concatenate([lon_vals - 0.125, [lon_vals[-1] + 0.125]])
ds_mass = ds_mass.assign_coords({'lat_b': lat_b, 'lon_b': lon_b})

# Attributes
ds_mass['lat'].attrs = {'units': 'degrees_north', 'standard_name': 'latitude'}
ds_mass['lon'].attrs = {'units': 'degrees_east', 'standard_name': 'longitude'}

# Masks
ds_land = xr.open_dataset(LAND_MASK_FILE)[['LO_val']].rename({'LO_val': 'mask_land'})
ds_ocean = xr.open_dataset(OCEAN_MASK_FILE)[['LO_val']].rename({'LO_val': 'mask_ocean'})

for ds in [ds_land, ds_ocean]:
    if 'latitude' in ds.coords: ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})

ds_source = xr.merge([ds_mass, ds_land, ds_ocean])
print("✅ Done.")

# =========================================================
# 3. REGRIDDING (SERIAL MODE)
# =========================================================
print("    Building Regridder (Single Core - Please Wait)...")
# This will take 5-10 minutes but will only use ~10-15GB RAM
regridder = xe.Regridder(
    ds_source, 
    ds_target, 
    method='conservative_normed', 
    periodic=True
)
print("✅ Regridder Built.")

print("    Regridding Data...")
# We apply the weights. Since Source is small, this is fast even on 1 core.
lwe_500m = regridder(ds_source['lwe']).astype('float32')
land_frac = regridder(ds_source['mask_land']).astype('float32')
ocean_frac = regridder(ds_source['mask_ocean']).astype('float32')
print("✅ Done.")

# =========================================================
# 4. SAVE
# =========================================================
print(f"    Saving to {OUTPUT_ZARR}...", end=" ")
ds_out = xr.Dataset({
    'lwe_thickness': lwe_500m,
    'grace_land_frac': land_frac,
    'grace_ocean_frac': ocean_frac
})

# Clear chunks encoding
for var in ds_out.variables:
    ds_out[var].encoding.pop('chunks', None)

# Write with Zarr chunks (Optimization for reading later)
ds_out = ds_out.chunk({'time': 1, 'y': 2048, 'x': 2048})
ds_out.to_zarr(OUTPUT_ZARR, mode='w', computed=True)

print("✅ SAVED.")
print(">>> 🏁 PHASE 1 (GRACE) COMPLETE.")
