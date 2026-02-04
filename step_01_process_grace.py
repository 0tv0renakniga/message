[200~import xarray as xr
  import numpy as np
  import os
  import dask.array as da
  from dask.distributed import Client
  from pyproj import Transformer
  import shutil

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

  # Cleanup previous failed runs
  if os.path.exists(OUTPUT_ZARR):
      shutil.rmtree(OUTPUT_ZARR)

  if __name__ == "__main__":
      # 1. SETUP CLIENT
      client = Client(n_workers=4, threads_per_worker=1, memory_limit='4GB')
      print(f">>> ⚡ Dask Client Running: {client.dashboard_link}")
      print(f">>> 🛰️ PROCESSING GRACE (Iterative Loop Strategy)...")

      # =========================================================
      # 2. PREPARE TARGET GRID
      # =========================================================
      print("    Preparing Target Grid...", end=" ")
      try:
          ds_target = xr.open_dataset(MASTER_GRID, engine='h5netcdf')
      except:
          ds_target = xr.open_dataset(MASTER_GRID)

      X, Y = np.meshgrid(ds_target['x'].values, ds_target['y'].values)
      transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
      lon_c, lat_c = transformer.transform(X, Y)

      # Convert to Dask Arrays (Chunked)
      lat_dask = da.from_array(lat_c, chunks=(2048, 2048))
      lon_dask = da.from_array(lon_c, chunks=(2048, 2048))

      ds_target['lat'] = (('y', 'x'), lat_dask)
      ds_target['lon'] = (('y', 'x'), lon_dask)
      ds_target = ds_target.set_coords(['lat', 'lon'])
      print("✅ Done.")

      # =========================================================
      # 3. PREPARE SOURCE DATA
      # =========================================================
      print("    Loading Source Data...", end=" ")
      
      # Load Mass (No time chunking, we iterate manually)
      ds_mass = xr.open_dataset(GRACE_MASS_FILE, engine='h5netcdf')[['lwe_thickness']]
      ds_mass = ds_mass.rename({'lwe_thickness': 'lwe'})

      # Masks
      ds_land = xr.open_dataset(LAND_MASK_FILE, chunks={}, engine='h5netcdf')[['LO_val']].rename({'LO_val': 'mask_land'})
      ds_ocean = xr.open_dataset(OCEAN_MASK_FILE, chunks={}, engine='h5netcdf')[['LO_val']].rename({'LO_val': 'mask_ocean'})

      # Merge Masks
      ds_masks = xr.merge([ds_land, ds_ocean])
      
      # FIX: Explicit Rename (No inplace=True)
      if 'latitude' in ds_mass.coords: 
          ds_mass = ds_mass.rename({'latitude': 'lat', 'longitude': 'lon'})
      
      if 'latitude' in ds_masks.coords: 
          ds_masks = ds_masks.rename({'latitude': 'lat', 'longitude': 'lon'})
      
      print("✅ Done.")

      # =========================================================
      # 4. PROCESS STATIC MASKS (Once)
      # =========================================================
      print("    Interpolating Static Masks...", end=" ")
      ds_masks_interp = ds_masks.interp(
              lat=ds_target['lat'], 
              lon=ds_target['lon'], 
              method='nearest'
          )
      
      ds_static_out = xr.Dataset({
              'grace_land_frac': ds_masks_interp['mask_land'],
              'grace_ocean_frac': ds_masks_interp['mask_ocean']
          })
      
      for var in ds_static_out.variables:
          ds_static_out[var].encoding.pop('chunks', None)
      ds_static_out = ds_static_out.chunk({'y': 2048, 'x': 2048})
      
      ds_static_out.to_zarr(OUTPUT_ZARR, mode='w', computed=True)
      print("✅ Masks Written.")

    # =========================================================
        # 5. PROCESS TIME STEPS (The Loop)
            # =========================================================
                print(f"    Processing {len(ds_mass.time)} Time Steps Iteratively...")
                    
                    times = ds_mass.time.values
                        
                        for i, t in enumerate(times):
                                    # A. Select Single Time Step
                                            ds_slice = ds_mass.sel(time=t)
                                                    
                                                    # B. Interpolate
                                                            interp_slice = ds_slice.interp(
                                                                        lat=ds_target['lat'], 
                                                                        lon=ds_target['lon'], 
                                                                        method='nearest'
                                                                    )
                                                                    
                                                                    # C. Expand dims
                                                                            ds_write = interp_slice.expand_dims('time')
                                                                                    
                                                                                    # D. Clean & Chunk
                                                                                            ds_write = ds_write.chunk({'time': 1, 'y': 2048, 'x': 2048})
                                                                                                    for var in ds_write.variables:
                                                                                                                    ds_write[var].encoding.pop('chunks', None)
                                                                                                                                
                                                                                                                            # E. Append
                                                                                                                                    # Note: 'grace_land_frac' etc are already written, so we just append 'lwe'
                                                                                                                                            # We need to make sure we only write 'lwe' to avoid overwriting masks
                                                                                                                                                    ds_write = ds_write[['lwe']]

                                                                                                                                                            if i == 0:
                                                                                                                                                                            # First write of 'lwe', use mode='a' to add to existing store
                                                                                                                                                                                        ds_write.to_zarr(OUTPUT_ZARR, mode='a', computed=True)
                                                                                                                                                                                                else:
                                                                                                          import xarray as xr
import numpy as np
import os
import dask.array as da
from dask.distributed import Client
from pyproj import Transformer
import shutil

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

# Cleanup previous failed runs
if os.path.exists(OUTPUT_ZARR):
    shutil.rmtree(OUTPUT_ZARR)

if __name__ == "__main__":
    # 1. SETUP CLIENT
    client = Client(n_workers=4, threads_per_worker=1, memory_limit='4GB')
    print(f">>> ⚡ Dask Client Running: {client.dashboard_link}")
    print(f">>> 🛰️ PROCESSING GRACE (Iterative Loop Strategy)...")

    # =========================================================
    # 2. PREPARE TARGET GRID
    # =========================================================
    print("    Preparing Target Grid...", end=" ")
    try:
        ds_target = xr.open_dataset(MASTER_GRID, engine='h5netcdf')
    except:
        ds_target = xr.open_dataset(MASTER_GRID)

    X, Y = np.meshgrid(ds_target['x'].values, ds_target['y'].values)
    transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    lon_c, lat_c = transformer.transform(X, Y)

    # Convert to Dask Arrays (Chunked)
    lat_dask = da.from_array(lat_c, chunks=(2048, 2048))
    lon_dask = da.from_array(lon_c, chunks=(2048, 2048))

    ds_target['lat'] = (('y', 'x'), lat_dask)
    ds_target['lon'] = (('y', 'x'), lon_dask)
    ds_target = ds_target.set_coords(['lat', 'lon'])
    print("✅ Done.")

    # =========================================================
    # 3. PREPARE SOURCE DATA
    # =========================================================
    print("    Loading Source Data...", end=" ")
    
    # Load Mass (No time chunking, we iterate manually)
    ds_mass = xr.open_dataset(GRACE_MASS_FILE, engine='h5netcdf')[['lwe_thickness']]
    ds_mass = ds_mass.rename({'lwe_thickness': 'lwe'})

    # Masks
    ds_land = xr.open_dataset(LAND_MASK_FILE, chunks={}, engine='h5netcdf')[['LO_val']].rename({'LO_val': 'mask_land'})
    ds_ocean = xr.open_dataset(OCEAN_MASK_FILE, chunks={}, engine='h5netcdf')[['LO_val']].rename({'LO_val': 'mask_ocean'})

    # Merge Masks
    ds_masks = xr.merge([ds_land, ds_ocean])
    
    # FIX: Explicit Rename (No inplace=True)
    if 'latitude' in ds_mass.coords: 
        ds_mass = ds_mass.rename({'latitude': 'lat', 'longitude': 'lon'})
    
    if 'latitude' in ds_masks.coords: 
        ds_masks = ds_masks.rename({'latitude': 'lat', 'longitude': 'lon'})
    
    print("✅ Done.")

    # =========================================================
    # 4. PROCESS STATIC MASKS (Once)
    # =========================================================
    print("    Interpolating Static Masks...", end=" ")
    ds_masks_interp = ds_masks.interp(
        lat=ds_target['lat'], 
        lon=ds_target['lon'], 
        method='nearest'
    )
    
    ds_static_out = xr.Dataset({
        'grace_land_frac': ds_masks_interp['mask_land'],
        'grace_ocean_frac': ds_masks_interp['mask_ocean']
    })
    
    for var in ds_static_out.variables:
        ds_static_out[var].encoding.pop('chunks', None)
    ds_static_out = ds_static_out.chunk({'y': 2048, 'x': 2048})
    
    ds_static_out.to_zarr(OUTPUT_ZARR, mode='w', computed=True)
    print("✅ Masks Written.")

    # =========================================================
    # 5. PROCESS TIME STEPS (The Loop)
    # =========================================================
    print(f"    Processing {len(ds_mass.time)} Time Steps Iteratively...")
    
    times = ds_mass.time.values
    
    for i, t in enumerate(times):
        # A. Select Single Time Step
        ds_slice = ds_mass.sel(time=t)
        
        # B. Interpolate
        interp_slice = ds_slice.interp(
            lat=ds_target['lat'], 
            lon=ds_target['lon'], 
            method='nearest'
        )
        
        # C. Expand dims
        ds_write = interp_slice.expand_dims('time')
        
        # D. Clean & Chunk
        ds_write = ds_write.chunk({'time': 1, 'y': 2048, 'x': 2048})
        for var in ds_write.variables:
            ds_write[var].encoding.pop('chunks', None)
            
        # E. Append
        # Note: 'grace_land_frac' etc are already written, so we just append 'lwe'
        # We need to make sure we only write 'lwe' to avoid overwriting masks
        ds_write = ds_write[['lwe']]

        if i == 0:
            # First write of 'lwe', use mode='a' to add to existing store
            ds_write.to_zarr(OUTPUT_ZARR, mode='a', computed=True)
        else:
            # Append subsequent times
            ds_write.to_zarr(OUTPUT_ZARR, mode='a', append_dim='time', computed=True)
            
        if i % 10 == 0:
            print(f"       Step {i}/{len(times)} complete...")

    print("✅ ALL TIME STEPS SAVED.")
    
    import zarr
    zarr.consolidate_metadata(OUTPUT_ZARR)
    print(">>> 🏁 PHASE 1 (GRACE) COMPLETE.")
    
    client.close()                                                                                                      # Append subsequent times
                                                                                                                                                                                                                            ds_write.to_zarr(OUTPUT_ZARR, mode='a', append_dim='time', computed=True)
                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                    if i % 10 == 0:
                                                                                                                                                                                                                                                    print(f"       Step {i}/{len(times)} complete...")

                                                                                                                                                                                                                                                        print("✅ ALL TIME STEPS SAVED.")
                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                            import zarr
                                                                                                                                                                                                                                                                zarr.consolidate_metadata(OUTPUT_ZARR)
                                                                                                                                                                                                                                                                    print(">>> 🏁 PHASE 1 (GRACE) COMPLETE.")
                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                        client.close()]
