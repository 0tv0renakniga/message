import xarray as xr
import xesmf as xe
import numpy as np
import os
import rioxarray  # <--- CRITICAL: Activates the .rio accessor
import cf_xarray  # <--- CRITICAL: Activates the .cf accessor (for bounds)

# --- CONFIGURATION ---
# UPDATE THESE PATHS to match your machine
RAW_DIR = "data/raw/grace"
PROC_DIR = "processed_layers"

# Files (Update filenames if yours differ slightly)
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

# 2. Generate 2D Lat/Lon for the Target
# xESMF needs 2D lat/lon arrays to calculate the overlap areas.
# We project our EPSG:3031 grid back to Lat/Lon just to get these coordinates.
print("    Generating 2D Target Mesh...", end=" ")
ds_target_mesh = ds_target.rio.reproject("EPSG:4326") 
ds_target['lat'] = ds_target_mesh.y
ds_target['lon'] = ds_target_mesh.x

# Add bounds (Critical for conservation!)
# This requires 'import cf_xarray' to work
ds_target = ds_target.cf.add_bounds(['lat', 'lon'])
print("✅ Done.")

# 3. Load Source Data (GRACE)
print("    Loading Raw GRACE Data...", end=" ")
# Load Mass, Land, Ocean
ds_mass = xr.open_dataset(GRACE_MASS_FILE)
ds_land = xr.open_dataset(LAND_MASK_FILE)
ds_ocean = xr.open_dataset(OCEAN_MASK_FILE)

# Merge them into one source dataset for cleaner processing
# We rename variables to be simpler/standard
ds_source = xr.merge([ds_mass, ds_land, ds_ocean])
ds_source = ds_source.rename({
    'lwe_thickness': 'lwe', 
    'land_mask': 'mask_land', 
    'ocean_mask': 'mask_ocean'
})

# Add bounds to source (Required for conservation)
ds_source = ds_source.cf.add_bounds(['lat', 'lon'])
print("✅ Done.")

# 4. Build the Regridder
print("    Building Conservative Regridder (This takes a moment)...")
# 'conservative_normed' is best for masking (handles coastlines better)
regridder = xe.Regridder(
    ds_source, 
    ds_target, 
    method='conservative_normed',
    periodic=True # GRACE is global, allows wrapping across the 180/-180 meridian
)

# 5. Execute Regridding
print("    Regridding Variables...", end=" ")
# Mass (Conserved)
lwe_500m = regridder(ds_source['lwe'])
# Masks (Become fractions 0.0-1.0)
land_frac = regridder(ds_source['mask_land'])
ocean_frac = regridder(ds_source['mask_ocean'])
print("✅ Done.")

# 6. Save to Zarr
print(f"    Saving to {OUTPUT_ZARR}...", end=" ")
ds_out = xr.Dataset({
    'lwe_thickness': lwe_500m,
    'grace_land_frac': land_frac,
    'grace_ocean_frac': ocean_frac
})

# Chunking is vital for Spark later. 
# We chunk strictly by Space (Time=1), so Spark can read one month of the whole continent.
ds_out.chunk({'time': 1, 'y': 2000, 'x': 2000}).to_zarr(OUTPUT_ZARR, mode='w')

print("✅ SAVED.")
print(">>> 🏁 PHASE 1 (GRACE) COMPLETE.")
