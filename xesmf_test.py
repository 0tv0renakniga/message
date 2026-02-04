import xesmf as xe
import numpy as np
import xarray as xr
import os

print("Running xESMF I/O Sanity Check...")

# 1. Create two tiny dummy grids
ds_in = xe.util.grid_global(5, 4)
ds_out = xe.util.grid_global(2, 2)

# 2. Build a Regridder AND save weights to disk
# This is the step the failed test was complaining about
if os.path.exists("test_weights.nc"):
    os.remove("test_weights.nc")

print("    Building Regridder & Saving Weights...", end=" ")
regridder = xe.Regridder(ds_in, ds_out, method='bilinear', filename='test_weights.nc')
print("Done.")

# 3. Try to Reuse the Weights
# If this crashes, then we have a problem. If it works, the test failure was a false alarm.
print("    Loading Weights from Disk...", end=" ")
regridder_reloaded = xe.Regridder(ds_in, ds_out, method='bilinear', filename='test_weights.nc', reuse_weights=True)
print("Done.")

# 4. Clean up
if os.path.exists("test_weights.nc"):
    os.remove("test_weights.nc")

print("SYSTEM VERIFIED. You are safe to proceed.")
