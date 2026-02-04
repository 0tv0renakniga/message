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

print(f">>> 🛰️ PROCESSING GRACE (Conservative Regridding)...")

# 1. Load the Master Grid (Target)
ds_target = xr.open_dataset(MASTER_GRID)

# CRITICAL FIX: Ensure CRS is set after loading
ds_target.rio.write_crs("EPSG:3031", inplace=True)

# 2. Generate 2D Lat/Lon Arrays (The Math Way)
# We do this manually to ensure dimensions match perfectly
print("    Calculating Target Coordinates...", end=" ")

# Create a 2D mesh of X and Y coordinates
X, Y = np.meshgrid(ds_target['x'], ds_target['y'])

# Transform from EPSG:3031 (Antarctica) to EPSG:4326 (Lat/Lon)
transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
lon_2d, lat_2d = transformer.transform(X, Y)

# Assign these 2D arrays back to the dataset
ds_target['lat'] = (('y', 'x'), lat_2d)
ds_target['lon'] = (('y', 'x'), lon_2d)

# Add bounds (Now works because we have valid 2D lat/lon)
ds_target = ds_target.cf.add_bounds(['lat', 'lon'])
print("✅ Done.")

# 3. Load Source Data (GRACE)
print("    Loading Raw GRACE Data...", end=" ")
ds_mass = xr.open_dataset(GRACE_MASS_FILE)
ds_land = xr.open_dataset(LAND_MASK_FILE)
ds_ocean = xr.open_dataset(OCEAN_MASK_FILE)

# Merge and Rename
ds_source = xr.merge([ds_mass, ds_land, ds_ocean])
ds_source = ds_source.rename({
    'lwe_thickness': 'lwe', 
    'land_mask': 'mask_land', 
    'ocean_mask': 'mask_ocean'
})

# Fix Source Bounds (GRACE is sometimes missing them or names them differently)
if 'lat_b' not in ds_source and 'lat_bounds' not in ds_source:
     ds_source = ds_source.cf.add_bounds(['lat', 'lon'])
print("✅ Done.")

# 4. Build the Regridder
print("    Building Conservative Regridder (This takes a moment)...")
regridder = xe.Regridder(
    ds_source, 
    ds_target, 
    method='conservative_normed',
    periodic=True
)

# 5. Execute Regridding
print("    Regridding Variables...", end=" ")
# The .astype('float32') saves disk space (GRACE precision doesn't need 64-bit)
lwe_500m = regridder(ds_source['lwe']).astype('float32')
land_frac = regridder(ds_source['mask_land']).astype('float32')
ocean_frac = regridder(ds_source['mask_ocean']).astype('float32')
print("✅ Done.")

# 6. Save to Zarr
print(f"    Saving to {OUTPUT_ZARR}...", end=" ")
ds_out = xr.Dataset({
    'lwe_thickness': lwe_500m,
    'grace_land_frac': land_frac,
    'grace_ocean_frac': ocean_frac
})

# Save chunked
ds_out.chunk({'time': 1, 'y': 2000, 'x': 2000}).to_zarr(OUTPUT_ZARR, mode='w')

print("✅ SAVED.")
print(">>> 🏁 PHASE 1 (GRACE) COMPLETE.")
