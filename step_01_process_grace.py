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

print(f">>> 🛰️ PROCESSING GRACE (Strict Compliance Fix)...")

# =========================================================
# 1. PREPARE TARGET GRID (2D Curvilinear)
# =========================================================
print("    Preparing Target Grid...", end=" ")

ds_target = xr.open_dataset(MASTER_GRID)

# A. Dummy Mask (Required for Parallel=True)
rows = ds_target.sizes['y']
cols = ds_target.sizes['x']
chunks = (2048, 2048)
ds_target['mask'] = (('y', 'x'), da.ones((rows, cols), chunks=chunks, dtype='int8'))

# B. Generate Coordinates (2D Centers)
X, Y = np.meshgrid(ds_target['x'].values, ds_target['y'].values)

# C. Generate Bounds (2D Corners, N+1 Shape)
# We use linspace to strictly enforce the N+1 shape (12289)
x_min = ds_target['x'].values[0] - 250.0
x_max = ds_target['x'].values[-1] + 250.0
y_max = ds_target['y'].values[0] + 250.0
y_min = ds_target['y'].values[-1] - 250.0

x_b = np.linspace(x_min, x_max, cols + 1)
y_b = np.linspace(y_max, y_min, rows + 1)
X_b, Y_b = np.meshgrid(x_b, y_b)

# D. Reproject to Lat/Lon
transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
lon_b, lat_b = transformer.transform(X_b, Y_b) # Corners
lon_c, lat_c = transformer.transform(X, Y)     # Centers

# E. Assign Variables (Simple Names)
ds_target['lat'] = (('y', 'x'), lat_c)
ds_target['lon'] = (('y', 'x'), lon_c)
ds_target['lat_b'] = (('y_b', 'x_b'), lat_b)
ds_target['lon_b'] = (('y_b', 'x_b'), lon_b)

# F. Set Attributes (The Fix for 'KeyError: Dataset.cf')
# We ONLY tag the centers. We leave bounds plain so cf_xarray doesn't get confused.
ds_target['lat'].attrs = {'units': 'degrees_north', 'standard_name': 'latitude'}
ds_target['lon'].attrs = {'units': 'degrees_east', 'standard_name': 'longitude'}

# G. Set Coordinates
ds_target = ds_target.set_coords(['lat', 'lon', 'lat_b', 'lon_b'])
print("✅ Done.")

# =========================================================
# 2. PREPARE SOURCE DATA (1D Rectilinear)
# =========================================================
print("    Loading Source Data...", end=" ")

# Load Mass (Lazy)
ds_mass = xr.open_dataset(GRACE_MASS_FILE, chunks={'time': 1})[['lwe_thickness']]
ds_mass = ds_mass.rename({'lwe_thickness': 'lwe'})

# Rename to simple lat/lon if needed
rename_dict = {}
if 'latitude' in ds_mass: rename_dict['latitude'] = 'lat'
if 'longitude' in ds_mass: rename_dict['longitude'] = 'lon'
ds_mass = ds_mass.rename(rename_dict)

# --- MANUAL BOUNDS (N+1) ---
# We calculate 1D bounds manually to ensure shape is (N+1)
# This fixes the "Assertion Error"
lat_vals = ds_mass.lat.values
lon_vals = ds_mass.lon.values

# Create edges (Size: 721 and 1441)
lat_b = np.concatenate([lat_vals - 0.125, [lat_vals[-1] + 0.125]])
lon_b = np.concatenate([lon_vals - 0.125, [lon_vals[-1] + 0.125]])

ds_mass = ds_mass.assign_coords({'lat_b': lat_b, 'lon_b': lon_b})

# Set Attributes (The Fix)
ds_mass['lat'].attrs = {'units': 'degrees_north', 'standard_name': 'latitude'}
ds_mass['lon'].attrs = {'units': 'degrees_east', 'standard_name': 'longitude'}
# Note: We do NOT add standard_name to lat_b/lon_b

# Load Masks
ds_land = xr.open_dataset(LAND_MASK_FILE, chunks={})[['LO_val']].rename({'LO_val': 'mask_land'})
ds_ocean = xr.open_dataset(OCEAN_MASK_FILE, chunks={})[['LO_val']].rename({'LO_val': 'mask_ocean'})

# Ensure masks match the naming
for ds in [ds_land, ds_ocean]:
    if 'latitude' in ds.coords: ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})

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
# Cast to float32
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

# Clear chunks encoding
for var in ds_out.variables:
    ds_out[var].encoding.pop('chunks', None)

# Write
ds_out = ds_out.chunk({'time': 1, 'y': 2048, 'x': 2048})
ds_out.to_zarr(OUTPUT_ZARR, mode='w', computed=True)

print("✅ SAVED.")
print(">>> 🏁 PHASE 1 (GRACE) COMPLETE.")
