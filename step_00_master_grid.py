import xarray as xr
import numpy as np
import rioxarray
import os

# Configuration
OUTPUT_DIR = "processed_layers"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("BUILDING MASTER GRID (EPSG:3031 @ 500m)...")

# 1. Define Bounds (Antarctic Polar Stereographic)
# These bounds cover the entire continent plus the shelf break
x_min, x_max = -3072000, 3072000
y_min, y_max = -3072000, 3072000
resolution = 500  # meters

# 2. Generate Coordinates (Center of Pixel)
# We use float64 to prevent the "floating point nightmare"
x = np.arange(x_min + resolution/2, x_max, resolution, dtype='float64')
y = np.arange(y_max - resolution/2, y_min, -resolution, dtype='float64')

# 3. Create the Template Dataset
ds_master = xr.Dataset(
    coords={
        'x': (['x'], x),
        'y': (['y'], y)
    }
)

# 4. Add CRS (Coordinate Reference System)
# This is critical for rioxarray to know "where" this grid is
ds_master.rio.write_crs("EPSG:3031", inplace=True)

# 5. Save
output_path = f"{OUTPUT_DIR}/master_grid_template.nc"
ds_master.to_netcdf(output_path)

print(f"Master Grid Saved: {output_path}")
print(f"Dimensions: {len(x)}x{len(y)}")
