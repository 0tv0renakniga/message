"""
ICESat-2 ATL15 Processing - SIMPLIFIED (No Quality Selection)
Just merge tiles and regrid to 500m
"""
import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe
import os
import shutil
import gc
from pyproj import Transformer
from dask.distributed import Client, LocalCluster

# ============================================================
# CONFIGURATION
# ============================================================
TEST_MODE = True
TEST_TIMESTEPS = 1
INPUT_DIR = "data/raw/icesat"
MASTER_GRID = "data/processed_layers/master_grid_template.nc"
OUTPUT_DIR = "data/processed_layers"

TILE_FILES = [
    f"{INPUT_DIR}/ATL15_A1_0328_01km_005_01.nc",
    f"{INPUT_DIR}/ATL15_A2_0328_01km_005_01.nc",
    f"{INPUT_DIR}/ATL15_A3_0328_01km_005_01.nc",
    f"{INPUT_DIR}/ATL15_A4_0328_01km_005_01.nc",
]

GROUPS = {
    'delta_h': {
        'vars': ['delta_h', 'delta_h_sigma', 'ice_area', 'data_count', 'misfit_rms'],
        'suffix': 'deltah'
    },
    'dhdt_lag1': {
        'vars': ['dhdt', 'dhdt_sigma', 'ice_area'],
        'suffix': 'lag1'
    },
    'dhdt_lag4': {
        'vars': ['dhdt', 'dhdt_sigma', 'ice_area'],
        'suffix': 'lag4'
    }
}

SOURCE_RESOLUTION = 1000
TARGET_RESOLUTION = 500
PIXEL_EDGE_BUFFER = SOURCE_RESOLUTION / 2

CHUNKS = {'time': 1, 'x': 1000, 'y': 1000}
OUTPUT_CHUNKS = {'time': 1, 'x': 2048, 'y': 2048}

# ============================================================
# FIX COORDINATES
# ============================================================
def fix_icesat2_coordinates(ds):
    """Fix coordinates to proper EPSG:3031 values"""
    pixel_size = SOURCE_RESOLUTION
    
    x_min = float(ds.x.min())
    x_max = float(ds.x.max())
    y_min = float(ds.y.min())
    y_max = float(ds.y.max())
    
    x_dim = len(ds.x)
    y_dim = len(ds.y)
    
    # Reconstruct coordinates
    x_coords = x_min + (np.arange(x_dim) * pixel_size)
    y_coords = y_max - (np.arange(y_dim) * pixel_size)  # Descending
    
    ds = ds.assign_coords({'x': x_coords, 'y': y_coords})
    
    ds.x.attrs.update({
        'long_name': 'x coordinate (polar stereographic)',
        'units': 'meter',
        'crs': 'EPSG:3031'
    })
    
    ds.y.attrs.update({
        'long_name': 'y coordinate (polar stereographic)',
        'units': 'meter',
        'crs': 'EPSG:3031'
    })
    
    return ds

# ============================================================
# DECODE TIME
# ============================================================
def decode_atl15_time(ds_group):
    """Decode ATL15 time (days since 2018-01-01)"""
    time_vals = ds_group['time'].values
    times_decoded = pd.to_datetime('2018-01-01') + pd.to_timedelta(time_vals, unit='D')
    return times_decoded

# ============================================================
# SIMPLE CONCATENATE - NO QUALITY SELECTION
# ============================================================
def concatenate_tiles_with_quality(tiles, quality_var='delta_h_sigma'):
    """
    Merge Antarctic tiles with quality-based selection at overlaps.
    Keep ALL coordinates - handle data conflicts intelligently.
    """
    if len(tiles) == 0:
        return None
    if len(tiles) == 1:
        return tiles[0]
    
    print(f"    Merging {len(tiles)} tiles with quality-based overlap handling...")
    
    # ================================================================
    # STEP 1: Sort coordinates in each tile to ensure monotonicity
    # ================================================================
    sorted_tiles = []
    for i, tile in enumerate(tiles):
        # Ensure X is monotonically increasing
        if not (tile.x.values[1:] > tile.x.values[:-1]).all():
            tile = tile.sortby('x')
        
        # Ensure Y is monotonically decreasing (standard for gridded data)
        if not (tile.y.values[1:] < tile.y.values[:-1]).all():
            tile = tile.sortby('y', ascending=False)
        
        sorted_tiles.append(tile)
        print(f"      Tile {i+1}: X=[{tile.x.min().values:,.0f}, {tile.x.max().values:,.0f}], "
              f"Y=[{tile.y.min().values:,.0f}, {tile.y.max().values:,.0f}]")
    
    # ================================================================
    # STEP 2: Identify overlapping regions
    # ================================================================
    tolerance = 10.0  # meters - coordinate matching tolerance
    
    # Find X overlaps (where tiles share X coordinates)
    x_boundary = 0.0
    has_x_overlap = any(
        np.any(np.abs(tile.x.values - x_boundary) < tolerance)
        for tile in sorted_tiles
    )
    
    # Find Y overlaps (where tiles share Y coordinates)  
    y_boundary = 0.0
    has_y_overlap = any(
        np.any(np.abs(tile.y.values - y_boundary) < tolerance)
        for tile in sorted_tiles
    )
    
    if has_x_overlap or has_y_overlap:
        print(f"      Overlaps detected: X={has_x_overlap}, Y={has_y_overlap}")
    
    # ================================================================
    # STEP 3: Merge with combine_by_coords (handles alignment)
    # ================================================================
    print(f"      Combining {len(sorted_tiles)} tiles...")
    
    # Use combine_by_coords which properly aligns coordinates
    merged = xr.combine_by_coords(
        sorted_tiles,
        compat='override',  # Allow data conflicts - we'll handle them
        combine_attrs='override'
    )
    
    print(f"      Initial merge: {dict(merged.sizes)}")
    
    # ================================================================
    # STEP 4: Quality-based selection at overlaps
    # ================================================================
    if (has_x_overlap or has_y_overlap) and quality_var in merged:
        print(f"      Applying quality-based selection using '{quality_var}'...")
        
        # Find pixels with data from multiple tiles
        # (These are where coordinates overlapped)
        data_count = sum(~np.isnan(tile[quality_var].values) for tile in sorted_tiles)
        overlap_mask = data_count > 1
        
        if np.any(overlap_mask):
            n_overlap = np.sum(overlap_mask)
            print(f"        Found {n_overlap:,} overlapping pixels")
            
            # At overlaps, keep data with LOWEST uncertainty
            # (lower sigma = higher quality)
            for var_name in merged.data_vars:
                if var_name == quality_var:
                    continue
                
                # Stack all tile versions
                all_versions = []
                all_quality = []
                
                for tile in sorted_tiles:
                    if var_name in tile:
                        # Align to merged grid
                        aligned = tile[var_name].reindex_like(merged[var_name])
                        all_versions.append(aligned.values)
                        
                        if quality_var in tile:
                            quality_aligned = tile[quality_var].reindex_like(merged[var_name])
                            all_quality.append(quality_aligned.values)
                
                if len(all_versions) > 1 and len(all_quality) > 0:
                    # Find version with best quality at each pixel
                    stacked_data = np.stack(all_versions, axis=0)
                    stacked_quality = np.stack(all_quality, axis=0)
                    
                    # Lower sigma = better, so use argmin
                    best_idx = np.nanargmin(stacked_quality, axis=0)
                    
                    # Select best data
                    best_data = np.take_along_axis(
                        stacked_data,
                        best_idx[np.newaxis, ...],
                        axis=0
                    )[0]
                    
                    merged[var_name].values = best_data
            
            print(f"        ✓ Quality selection complete")
    
    # ================================================================
    # STEP 5: Validation
    # ================================================================
    x_unique = len(np.unique(merged.x.values))
    y_unique = len(np.unique(merged.y.values))
    
    if x_unique != len(merged.x):
        raise ValueError(f"Duplicate X coordinates remain: {len(merged.x)} vs {x_unique}")
    if y_unique != len(merged.y):
        raise ValueError(f"Duplicate Y coordinates remain: {len(merged.y)} vs {y_unique}")
    
    # Check monotonicity
    x_increasing = (merged.x.values[1:] > merged.x.values[:-1]).all()
    y_decreasing = (merged.y.values[1:] < merged.y.values[:-1]).all()
    
    if not x_increasing:
        raise ValueError("X coordinates not monotonically increasing")
    if not y_decreasing:
        raise ValueError("Y coordinates not monotonically decreasing")
    
    print(f"\n      ✓ Final grid: {dict(merged.sizes)}")
    print(f"        X: [{merged.x.min().values:,.0f}, {merged.x.max().values:,.0f}] (monotonic ↑)")
    print(f"        Y: [{merged.y.min().values:,.0f}, {merged.y.max().values:,.0f}] (monotonic ↓)")
    
    # Check data coverage
    first_var = list(merged.data_vars)[0]
    if 'time' in merged.dims:
        valid = np.sum(~np.isnan(merged[first_var].isel(time=0).values))
        total = merged[first_var].isel(time=0).size
    else:
        valid = np.sum(~np.isnan(merged[first_var].values))
        total = merged[first_var].size
    
    coverage = 100 * valid / total
    print(f"        Coverage: {coverage:.1f}% ({valid:,} valid pixels)")
    
    if coverage < 1.0:
        print(f"        ⚠️  WARNING: Very low coverage - check if data was lost!")
    
    return merged

# ============================================================
# REGRID TO 500M
# ============================================================
def regrid_1000m_to_500m(ds_source, target_grid, weight_file='weights_1000m_to_500m.nc'):
    """Regrid from 1000m to 500m"""
    print(f"    Regridding 1000m → 500m...")
    
    # Subset target grid to source extent
    source_x_min = ds_source.x.min().values - PIXEL_EDGE_BUFFER
    source_x_max = ds_source.x.max().values + PIXEL_EDGE_BUFFER
    source_y_min = ds_source.y.min().values - PIXEL_EDGE_BUFFER
    source_y_max = ds_source.y.max().values + PIXEL_EDGE_BUFFER
    
    buffer = SOURCE_RESOLUTION
    target_x_mask = (target_grid.x >= source_x_min - buffer) & (target_grid.x <= source_x_max + buffer)
    target_y_mask = (target_grid.y >= source_y_min - buffer) & (target_grid.y <= source_y_max + buffer)
    
    x_indices = np.where(target_x_mask)[0]
    y_indices = np.where(target_y_mask)[0]
    
    target_x_subset = target_grid.x.values[x_indices]
    target_y_subset = target_grid.y.values[y_indices]
    
    target_subset = xr.Dataset({
        'lon': (('y', 'x'), target_grid.lon.values[np.ix_(y_indices, x_indices)]),
        'lat': (('y', 'x'), target_grid.lat.values[np.ix_(y_indices, x_indices)]),
    }, coords={'x': target_x_subset, 'y': target_y_subset})
    
    print(f"      Source: {ds_source.sizes['y']} x {ds_source.sizes['x']}")
    print(f"      Target: {len(target_y_subset)} x {len(target_x_subset)}")
    
    # Create regridder
    reuse = os.path.exists(weight_file)
    regridder = xe.Regridder(
        ds_source, target_subset, 'bilinear',
        reuse_weights=reuse, filename=weight_file, ignore_degenerate=True
    )
    
    # Regrid
    ds_regrid = regridder(ds_source, keep_attrs=True)
    ds_regrid = ds_regrid.compute()
    
    # Check coverage
    first_var = list(ds_regrid.data_vars)[0]
    valid = np.sum(~np.isnan(ds_regrid[first_var].values))
    total = ds_regrid[first_var].size
    coverage = 100 * valid / total
    print(f"      ✓ Coverage: {coverage:.1f}%")
    
    del regridder
    gc.collect()
    
    return ds_regrid

# ============================================================
# PROCESS GROUP - SIMPLIFIED
# ============================================================
def process_group(group_name, group_config, target_grid, transformer, times):
    """Process one variable group - SIMPLIFIED WORKFLOW"""
    
    print(f"\n{'='*60}")
    print(f"PROCESSING: {group_name.upper()}")
    print(f"{'='*60}")
    
    vars_to_process = group_config['vars']
    suffix = group_config['suffix']
    suffix_full = f"{suffix}_test" if TEST_MODE else suffix
    output_zarr = f"{OUTPUT_DIR}/icesat2_500m_{suffix_full}.zarr"
    weight_file = f"weights_{group_name}_1000m_to_500m.nc"
    
    print(f"Variables: {vars_to_process}")
    print(f"Output: {output_zarr}\n")
    
    # Cleanup
    if os.path.exists(output_zarr):
        shutil.rmtree(output_zarr)
    
    # [1/3] LOAD TILES
    print(f"[1/3] Loading {len(TILE_FILES)} tiles...")
    tiles = []
    
    for i, tile_file in enumerate(TILE_FILES):
        print(f"    Tile {i+1}: {os.path.basename(tile_file)}")
        
        ds = xr.open_dataset(tile_file, group=group_name, chunks=CHUNKS, decode_times=False)
        ds = fix_icesat2_coordinates(ds)
        times_decoded = decode_atl15_time(ds)
        ds = ds.assign_coords(time=times_decoded)
        
        ds_subset = ds[vars_to_process]
        tiles.append(ds_subset)
        print(f"      ✓ {dict(ds_subset.sizes)}")
    
    # [2/3] SIMPLE CONCATENATE
    print(f"\n[2/3] Merging tiles...")
    merged = concatenate_tiles_with_quality(tiles, quality_var='delta_h_sigma')
    
    # Add lon/lat
    print(f"\n    Adding lon/lat...")
    X, Y = np.meshgrid(merged.x.values, merged.y.values)
    lons, lats = transformer.transform(X, Y)
    merged['lon'] = (('y', 'x'), lons)
    merged['lat'] = (('y', 'x'), lats)
    
    # [3/3] REGRID AND SAVE
    print(f"\n[3/3] Processing {len(times)} timesteps...")
    
    for t_idx, time_val in enumerate(times):
        print(f"\n  Timestep {t_idx+1}/{len(times)}: {time_val}")
        
        ds_t = merged.isel(time=t_idx)
        ds_regrid = regrid_1000m_to_500m(ds_t, target_grid, weight_file)
        
        ds_regrid = ds_regrid.expand_dims(time=[time_val])
        ds_regrid = ds_regrid.chunk(OUTPUT_CHUNKS)
        
        if t_idx == 0:
            ds_regrid.to_zarr(output_zarr, mode='w')
        else:
            ds_regrid.to_zarr(output_zarr, append_dim='time')
        
        print(f"  ✓ Saved")
        del ds_regrid, ds_t
        gc.collect()
    
    print(f"\n✓ Complete: {output_zarr}\n")
    del merged, tiles
    gc.collect()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    
    # Verify files
    print("\n" + "="*60)
    print("FILE VERIFICATION")
    print("="*60 + "\n")
    
    for tile_file in TILE_FILES:
        exists = os.path.exists(tile_file)
        status = "✓" if exists else "✗ MISSING"
        size_gb = os.path.getsize(tile_file) / 1e9 if exists else 0
        print(f"{status} {os.path.basename(tile_file)} ({size_gb:.1f} GB)")
        if not exists:
            exit(1)
    
    if not os.path.exists(MASTER_GRID):
        print(f"\n❌ ERROR: Master grid not found")
        exit(1)
    
    print(f"✓ Master grid: {MASTER_GRID}\n")
    
    # Dask
    print("="*60)
    print("DASK CLIENT")
    print("="*60 + "\n")
    
    cluster = LocalCluster(n_workers=8, threads_per_worker=4, memory_limit='16GB', silence_logs=True)
    client = Client(cluster)
    print(f"✓ Dashboard: {client.dashboard_link}\n")
    
    # Initialize
    print("="*60)
    print("INITIALIZATION")
    print("="*60 + "\n")
    
    ds_sample = xr.open_dataset(TILE_FILES[0], group='delta_h', decode_times=False)
    times = decode_atl15_time(ds_sample)
    
    if TEST_MODE:
        times = times[:TEST_TIMESTEPS]
        print(f"🧪 TEST MODE: {TEST_TIMESTEPS} timestep(s)")
    
    print(f"Timesteps: {len(times)}")
    ds_sample.close()
    
    ds_target = xr.open_dataset(MASTER_GRID)
    x_500m = ds_target.x.values
    y_500m = ds_target.y.values
    print(f"Target grid: {len(y_500m)} x {len(x_500m)}")
    
    transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    X_target, Y_target = np.meshgrid(x_500m, y_500m)
    lons_target, lats_target = transformer.transform(X_target, Y_target)
    
    target_grid = xr.Dataset({
        'lon': (('y', 'x'), lons_target),
        'lat': (('y', 'x'), lats_target),
        'x': x_500m,
        'y': y_500m
    })
    print(f"✓ Target grid ready\n")
    
    # Process
    print("="*60)
    print("PROCESSING")
    print("="*60)
    
    for group_name, group_config in GROUPS.items():
        try:
            process_group(group_name, group_config, target_grid, transformer, times)
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("✅ COMPLETE!")
    print("="*60 + "\n")
    
    for group_name, group_config in GROUPS.items():
        suffix_full = f"{group_config['suffix']}_test" if TEST_MODE else group_config['suffix']
        output_path = f"{OUTPUT_DIR}/icesat2_500m_{suffix_full}.zarr"
        
        if os.path.exists(output_path):
            ds_check = xr.open_zarr(output_path)
            print(f"✓ {output_path}")
            print(f"  {dict(ds_check.sizes)}")
            print(f"  {list(ds_check.data_vars)}\n")
            ds_check.close()
    
    client.close()
    cluster.close()
    print("🎉 Done!")

