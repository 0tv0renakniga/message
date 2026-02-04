import xarray as xr
import xesmf as xe
import numpy as np
import os

# --- PATHS ---
# UPDATE THESE to point to your actual downloaded files
GRACE_MASS_FILE = "data/raw/grace/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections_v02.nc"
LAND_MASK_FILE  = "data/raw/grace/CSR_GRACE_GRACE-FO_RL06_Mascons_LandMask.nc"
OCEAN_MASK_FILE = "data/raw/grace/CSR_GRACE_GRACE-FO_RL06_Mascons_OceanMask.nc"
MASTER_GRID     = "processed_layers/master_grid_template.nc"
OUTPUT_ZARR     = "processed_layers/grace_500m.zarr"

print("PROCESSING GRACE (Conservative Regridding)...")

# 1. Load the Master Grid (Target)
ds_target = xr.open_dataset(MASTER_GRID)

# 2. Generate 2D Lat/Lon for the Target
# xESMF needs 2D lat/lon arrays to calculate the overlap areas
print("Generating 2D Target Mesh...", end=" ")
ds_target_mesh = ds_target.rio.reproject("EPSG:4326") # Temporarily project to Lat/Lon to get coords
ds_target['lat'] = ds_target_mesh.y
ds_target['lon'] = ds_target_mesh.x
# Add bounds (Critical for conservation!)
ds_target = ds_target.cf.add_bounds(['lat', 'lon'])
print("Done.")

# 3. Load Source Data (GRACE)
print("Loading Raw GRACE Data...", end=" ")
# Load Mass, Land, Ocean
ds_mass = xr.open_dataset(GRACE_MASS_FILE)
ds_land = xr.open_dataset(LAND_MASK_FILE)
ds_ocean = xr.open_dataset(OCEAN_MASK_FILE)

# Merge them into one source dataset for cleaner processing
ds_source = xr.merge([ds_mass, ds_land, ds_ocean])
ds_source = ds_source.rename({'lwe_thickness': 'lwe', 'land_mask': 'mask_land', 'ocean_mask': 'mask_ocean'})

# Add bounds to source (Required for conservation)
ds_source = ds_source.cf.add_bounds(['lat', 'lon'])
print("Done.")

# 4. Build the Regridder
print("Building Conservative Regridder (This takes a moment)...")
regridder = xe.Regridder(
    ds_source, 
    ds_target, 
    method='conservative_normed',
    periodic=True # GRACE is global, so it wraps around the longitude
)

# 5. Execute Regridding
print("Regridding Variables...", end=" ")
# Mass (Conserved)
lwe_500m = regridder(ds_source['lwe'])
# Masks (Become fractions 0.0-1.0)
land_frac = regridder(ds_source['mask_land'])
ocean_frac = regridder(ds_source['mask_ocean'])
print("Done.")

# 6. Save to Zarr
print(f"    Saving to {OUTPUT_ZARR}...", end=" ")
ds_out = xr.Dataset({
    'lwe_thickness': lwe_500m,
    'grace_land_frac': land_frac,
    'grace_ocean_frac': ocean_frac
})

# Chunking is vital for Spark later. 
# We chunk strictly by Space (Time=1), so Spark can read one month of the whole continent easily.
ds_out.chunk({'time': 1, 'y': 2000, 'x': 2000}).to_zarr(OUTPUT_ZARR, mode='w')
print("SAVED.")

print("PHASE 1 (GRACE) COMPLETE.")
