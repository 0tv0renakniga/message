import xarray as xr
import pandas as pd
import xesmf as xe
import numpy as np
import os
import shutil
from dask.distributed import Client
from pyproj import Transformer
import dask.array as da
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# --- CONFIGURATION ---
OCEAN_MASK_FILE = "data/raw/grace/CSR_GRACE_GRACE-FO_RL06_Mascons_v02_OceanMask.nc"
LAND_MASK_FILE  = "data/raw/grace/CSR_GRACE_GRACE-FO_RL06_Mascons_v02_LandMask.nc"
GRACE_MASS_FILE = "data/raw/grace/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc"
MASTER_GRID     = "data/processed_layers/master_grid_template.nc"
OUTPUT_ZARR     = "data/processed_layers/grace_500m.zarr"

# Weight files
WEIGHTS_MASKS = "weights_masks.nc"
WEIGHTS_MAIN  = "weights_grace_main.nc"

# Memory tuned for your data
TEMPORAL_BATCH = 1
SPATIAL_TILE_Y = 2048
SPATIAL_TILE_X = 2048
ZARR_CHUNKS = (1, 2048, 2048)

# QC THRESHOLDS
QC_MIN = -5000  # cm
QC_MAX = 5000   # cm

# Cleanup
if os.path.exists(OUTPUT_ZARR):
    shutil.rmtree(OUTPUT_ZARR)

if __name__ == "__main__":
    
    # 1. DASK CLIENT
    client = Client(
        n_workers=8,
        threads_per_worker=1,
        memory_limit='6GB',
        dashboard_address=':8787',
        processes=True
    )
    print(f">>> Dask: {client.dashboard_link}\n")

    # 2. LOAD TARGET GRID
    print("[1/6] Loading target grid...")
    ds_target = xr.open_dataset(MASTER_GRID)
    
    x_coords = ds_target['x'].values
    y_coords = ds_target['y'].values
    ny, nx = len(y_coords), len(x_coords)
    
    print(f"    Target: {ny} x {nx} pixels")
    print(f"    Target size per time step: {ny * nx * 4 / 1e6:.0f} MB")
    # Estimate memory per chunk
    for chunk_size in [500, 1000, 1500, 2000, 3000, 4000]:
        n_chunks = (ny // chunk_size + 1) * (nx // chunk_size + 1)
        chunk_pixels = chunk_size * chunk_size
        print(f"    Chunk {chunk_size:4d}: {n_chunks:3d} chunks, "
            f"{chunk_pixels/1e6:.1f}M pixels/chunk, "
            f"~{chunk_pixels*4/1e6:.0f} MB data/chunk")
    # Calculate lat/lon for target
    print("    Computing target lat/lon...")
    X, Y = np.meshgrid(x_coords, y_coords)
    transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    target_lons, target_lats = transformer.transform(X, Y)
    print("    Done.\n")

    # 3. PROCESS STATIC MASKS
    print("[2/6] Processing masks...")
    
    ds_land = xr.open_dataset(LAND_MASK_FILE)[['LO_val']].rename(
        {'LO_val': 'mask_land'}
    ).load()
    
    ds_ocean = xr.open_dataset(OCEAN_MASK_FILE)[['LO_val']].rename(
        {'LO_val': 'mask_ocean'}
    ).load()
    
    ds_masks = xr.merge([ds_land, ds_ocean])
    
    # Create target grid for regridder
    ds_target_grid = xr.Dataset({
        'lon': (('y', 'x'), target_lons),
        'lat': (('y', 'x'), target_lats)
    })
    
    # Build mask regridder - USE NEAREST FOR MASKS
    print("    Building mask regridder (nearest_s2d)...")
    reuse_mask_weights = os.path.exists(WEIGHTS_MASKS)
    
    regridder_masks = xe.Regridder(
        ds_masks,
        ds_target_grid,
        'nearest_s2d',  # MASKS DON'T NEED CONSERVATIVE
        reuse_weights=reuse_mask_weights,
        filename=WEIGHTS_MASKS
    )
    
    # Regrid masks
    print("    Regridding masks...")
    ds_masks_regrid = regridder_masks(ds_masks, keep_attrs=True)
    
    # Convert to numpy and save
    mask_land_data = ds_masks_regrid['mask_land'].values
    mask_ocean_data = ds_masks_regrid['mask_ocean'].values
    
    # Create static dataset
    ds_static = xr.Dataset(
        {
            'grace_land_frac': (('y', 'x'), mask_land_data),
            'grace_ocean_frac': (('y', 'x'), mask_ocean_data)
        },
        coords={'y': y_coords, 'x': x_coords}
    )
    
    # Chunk and save
    ds_static = ds_static.chunk({'y': SPATIAL_TILE_Y, 'x': SPATIAL_TILE_X})
    ds_static.to_zarr(OUTPUT_ZARR, mode='w')
    print("    Masks saved.\n")
    
    del ds_masks_regrid, mask_land_data, mask_ocean_data

    # 4. LOAD SOURCE DATA
    print("[3/6] Opening GRACE dataset...")
    # Load and decode time properly

    from datetime import timedelta

    # Open WITHOUT auto-decoding
    ds_raw = xr.open_dataset(GRACE_MASS_FILE, decode_times=False)

    print(f"    Raw time values: {ds_raw.time.values[0]} to {ds_raw.time.values[-1]}")
    print(f"    Total timesteps: {len(ds_raw.time)}")

    # MANUAL TIME CONVERSION - Using native timedelta
    reference_date = pd.Timestamp("2002-01-01")
    time_decoded = [reference_date + timedelta(days=float(t)) for t in ds_raw['time'].values]

    print(f"\n    Decoded time range:")
    print(f"      First: {time_decoded[0]}")
    print(f"      Last: {time_decoded[-1]}")

    # Create dataset with properly decoded time
    ds_mass = ds_raw[['lwe_thickness']].assign_coords(time=time_decoded)
    ds_mass = ds_mass.rename({'lwe_thickness': 'lwe'})

    # NOW filter to 2019-2025 (using ds_mass, NOT ds!)
    ds_mass = ds_mass.sel(time=slice('2019-01-01', '2025-12-31'))
    ds_mass = ds_mass.chunk({'time': 1})

    print(f"\n    Filtered to 2019-2025:")
    print(f"      Timesteps: {len(ds_mass.time)}")
    if len(ds_mass.time) > 0:
        print(f"      First: {pd.Timestamp(ds_mass.time.values[0]).strftime('%Y-%m')}")
        print(f"      Last: {pd.Timestamp(ds_mass.time.values[-1]).strftime('%Y-%m')}")
        print(f"      Source resolution: {len(ds_mass.lat)} x {len(ds_mass.lon)}")

    total_times = len(ds_mass.time)
 
    # 5. BUILD REGRIDDER
    print("[4/6] Building main regridder...")
    reuse_main_weights = os.path.exists(WEIGHTS_MAIN)
    
    # For GRACE data: bilinear is fine and much faster
    # If you MUST use conservative, see note below
    regridder = xe.Regridder(
        ds_mass.isel(time=0).drop_vars('time'),
        ds_target_grid,
        'nearest_s2d',  # FAST, LOW MEMORY
        reuse_weights=reuse_main_weights,
        filename=WEIGHTS_MAIN
    )
    print("    Regridder ready.\n")

    # 6. PRE-ALLOCATE ZARR
    print("[5/6] Pre-allocating Zarr store...")
    
    lwe_data = da.empty(
        (total_times, ny, nx),
        chunks=ZARR_CHUNKS,
        dtype=np.float32
    )
    
    ds_template = xr.Dataset(
        {'lwe': (('time', 'y', 'x'), lwe_data)},
        coords={
            'time': ds_mass.time.values,
            'y': y_coords,
            'x': x_coords
        }
    )
    
    # ADD QC METADATA TO TEMPLATE
    ds_template['lwe'].attrs['qc_applied'] = 'outliers_masked'
    ds_template['lwe'].attrs['qc_range'] = f'({QC_MIN}, {QC_MAX}) cm'
    ds_template['lwe'].attrs['qc_reason'] = 'Remove coastal/boundary interpolation artifacts'
    
    ds_template.to_zarr(OUTPUT_ZARR, mode='a', compute=False)
    print(f"    Allocated {total_times} time steps.\n")

    # 7. PROCESS TIME STEPS WITH SPATIAL TILING + QC
    print(f"[6/6] Processing {total_times} time steps with QC masking...")
    print(f"    Strategy: 1 time step x spatial tiles ({SPATIAL_TILE_Y}x{SPATIAL_TILE_X})")
    print(f"    QC Range: {QC_MIN} to {QC_MAX} cm\n")
    
    # Track QC stats
    total_pixels_masked = 0
    total_pixels_processed = 0
    
    for t_idx in range(total_times):
        print(f"    Time {t_idx+1}/{total_times}:", end=" ")
        
        # Load ONE time step
        ds_single = ds_mass.isel(time=t_idx).load()
        
        # Regrid
        ds_regridded = regridder(ds_single, keep_attrs=True)
        
        # ============================================================
        # APPLY QUALITY CONTROL MASK
        # ============================================================
        # Count valid pixels before QC
        data_before_qc = ds_regridded['lwe'].values
        valid_before = ~np.isnan(data_before_qc)
        
        # Apply mask: Replace outliers with NaN
        ds_regridded['lwe'] = ds_regridded['lwe'].where(
            (ds_regridded['lwe'] > QC_MIN) & (ds_regridded['lwe'] < QC_MAX),
            drop=False
        )
        
        # Count what was masked
        data_after_qc = ds_regridded['lwe'].values
        valid_after = ~np.isnan(data_after_qc)
        masked_this_step = valid_before.sum() - valid_after.sum()
        
        total_pixels_masked += masked_this_step
        total_pixels_processed += valid_before.sum()
        # ============================================================
        
        # Convert to numpy
        data_regridded = ds_regridded['lwe'].values
        
        # Write in spatial tiles
        for y_start in range(0, ny, SPATIAL_TILE_Y):
            y_end = min(y_start + SPATIAL_TILE_Y, ny)
            
            for x_start in range(0, nx, SPATIAL_TILE_X):
                x_end = min(x_start + SPATIAL_TILE_X, nx)
                
                # Extract tile
                tile_data = data_regridded[y_start:y_end, x_start:x_end]
                
                # Create mini dataset for this tile
                ds_tile = xr.Dataset(
                    {'lwe': (('time', 'y', 'x'), tile_data[np.newaxis, :, :])},
                    coords={
                        'time': [ds_mass.time.values[t_idx]],  # Single timestep from GRACE
                        'y': y_coords[y_start:y_end],          # From master grid
                        'x': x_coords[x_start:x_end]           # From master grid
                    }
                )
                
                # Write to specific region
                ds_tile.to_zarr(
                    OUTPUT_ZARR,
                    region={
                        'time': slice(t_idx, t_idx + 1),
                        'y': slice(y_start, y_end),
                        'x': slice(x_start, x_end)
                    }
                )
        
        # Show masked count for this timestep if any
        if masked_this_step > 0:
            print(f"Done ({(t_idx+1)/total_times*100:.0f}%) [Masked: {masked_this_step:,} pixels]")
        else:
            print(f"Done ({(t_idx+1)/total_times*100:.0f}%)")
        
        # Clean up
        del ds_single, ds_regridded, data_regridded

    # FINAL QC REPORT
    print("\n" + "=" * 60)
    print("QUALITY CONTROL SUMMARY")
    print("=" * 60)
    print(f"  Total pixels processed: {total_pixels_processed:,}")
    print(f"  Total pixels masked:    {total_pixels_masked:,}")
    pct_masked = (total_pixels_masked / total_pixels_processed * 100) if total_pixels_processed > 0 else 0
    print(f"  Percentage masked:      {pct_masked:.4f}%")
    print(f"  QC range applied:       {QC_MIN} to {QC_MAX} cm")
    print("=" * 60)

    # CLEANUP
    print("\n>>> COMPLETE!")
    print(f">>> Output: {OUTPUT_ZARR}")
    print(f">>> Size: {total_times * ny * nx * 4 / 1e9:.1f} GB")
    
    client.close()
