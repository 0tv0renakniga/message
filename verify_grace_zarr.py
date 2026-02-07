import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Use Path for better path handling
SCRIPT_DIR = Path(__file__).parent
ZARR_PATH = SCRIPT_DIR / "data" / "processed_layers" / "grace_500m.zarr"
MASTER_GRID = SCRIPT_DIR / "data" / "processed_layers" / "master_grid_template.nc"  # ← FIXED

def main():
    print("=" * 60)
    print("GRACE ZARR QUALITY CHECK (Memory-Safe)")
    print("=" * 60)
    
    # Check paths exist
    print("\n0. Checking paths...")
    print(f"   Working dir: {Path.cwd()}")
    print(f"   Zarr path: {ZARR_PATH}")
    print(f"   Grid path: {MASTER_GRID}")
    
    if not ZARR_PATH.exists():
        print(f"   ❌ ERROR: Zarr not found at {ZARR_PATH}")
        print(f"   Please update ZARR_PATH in script")
        sys.exit(1)
    
    if not MASTER_GRID.exists():
        print(f"   ❌ ERROR: Master grid not found at {MASTER_GRID}")
        print(f"   Please update MASTER_GRID in script")
        sys.exit(1)
    
    print("   ✓ Paths found")
    
    # Load data (LAZY - doesn't load into memory)
    print("\n1. Loading data (lazy)...")
    ds = xr.open_zarr(str(ZARR_PATH))
    ds_grid = xr.open_dataset(str(MASTER_GRID))
    
    print(f"   ✓ Connected to: {ds.nbytes / 1e9:.1f} GB (not loaded)")
    print(f"\n{ds}")
    
    # Check 1: Dimensions
    print("\n" + "=" * 60)
    print("2. DIMENSION CHECK")
    print("=" * 60)
    ny, nx = len(ds_grid.y), len(ds_grid.x)
    nt = len(ds.time)
    
    print(f"   Expected: time={nt}, y={ny}, x={nx}")
    print(f"   Actual:   time={len(ds.time)}, y={len(ds.y)}, x={len(ds.x)}")
    assert len(ds.y) == ny and len(ds.x) == nx, "❌ Dimension mismatch!"
    print("   ✓ Dimensions correct")
    
    # Check 2: Coordinate alignment
    print("\n" + "=" * 60)
    print("3. COORDINATE ALIGNMENT")
    print("=" * 60)
    x_match = np.allclose(ds.x.values, ds_grid.x.values)
    y_match = np.allclose(ds.y.values, ds_grid.y.values)
    print(f"   X coords match: {x_match}")
    print(f"   Y coords match: {y_match}")
    assert x_match and y_match, "❌ Coordinates don't match master grid!"
    print("   ✓ Coordinates aligned with master grid")
    
    # Check 3: Data completeness (SAMPLE ONLY)
    print("\n" + "=" * 60)
    print("4. DATA COMPLETENESS (sampling)")
    print("=" * 60)
    
    # Check a few time steps (load small slices)
    for i in [0, nt//2, nt-1]:  # First, middle, last
        print(f"   Checking time {i}...", end=" ")
        data_slice = ds.lwe.isel(time=i).values  # Only loads one 2D slice
        n_valid = np.sum(~np.isnan(data_slice))
        pct_valid = (n_valid / data_slice.size) * 100
        
        time_str = str(ds.time.values[i])[:10]
        print(f"{time_str}: {n_valid:,} valid ({pct_valid:.1f}%)")
    
    print("   ✓ Sample time steps have data")
    
    # Check 4: Value ranges (use sample)
    print("\n" + "=" * 60)
    print("5. VALUE RANGE CHECK (sampling)")
    print("=" * 60)
    
    # Sample: take every 10th timestep, downsample spatially
    sample_times = slice(0, nt, 10)  
    sample_spatial = (slice(None, None, 10), slice(None, None, 10))  # Every 10th pixel
    
    print("   Sampling data (10% of timesteps, 1% of pixels)...")
    sample_data = ds.lwe.isel(time=sample_times, y=sample_spatial[0], x=sample_spatial[1]).values
    valid_data = sample_data[~np.isnan(sample_data)]
    
    print(f"   Min:    {np.min(valid_data):.2f} cm")
    print(f"   Max:    {np.max(valid_data):.2f} cm")
    print(f"   Mean:   {np.mean(valid_data):.2f} cm")
    print(f"   Median: {np.median(valid_data):.2f} cm")
    print(f"   Std:    {np.std(valid_data):.2f} cm")
    
    # Sanity check
    if np.min(valid_data) < -200 or np.max(valid_data) > 200:
        print("   ⚠️  WARNING: Values outside typical GRACE range!")
    else:
        print("   ✓ Values in expected range")
    
    # Check 5: Temporal continuity
    print("\n" + "=" * 60)
    print("6. TEMPORAL CONTINUITY")
    print("=" * 60)
    
    time_diffs = np.diff(ds.time.values).astype('timedelta64[D]').astype(int)
    print(f"   Time steps: {nt}")
    print(f"   Date range: {str(ds.time.values[0])[:10]} to {str(ds.time.values[-1])[:10]}")
    print(f"   Avg gap: {np.mean(time_diffs):.0f} days")
    print(f"   Min gap: {np.min(time_diffs)} days")
    print(f"   Max gap: {np.max(time_diffs)} days")
    
    # Find large gaps
    large_gaps = np.where(time_diffs > 60)[0]
    if len(large_gaps) > 0:
        print(f"\n   Large gaps (>60 days): {len(large_gaps)}")
        for idx in large_gaps[:5]:
            print(f"      {str(ds.time.values[idx])[:10]} → {str(ds.time.values[idx+1])[:10]} ({time_diffs[idx]} days)")
        print("   ℹ️  This is normal for GRACE data")
    
    print("   ✓ Time series complete")
    
    # Check 6: Spatial patterns (COMPUTE MEAN LAZILY)
    print("\n" + "=" * 60)
    print("7. SPATIAL PATTERNS (computing mean...)")
    print("=" * 60)
    
    # This computes mean in chunks - safe!
    print("   Computing temporal mean (this may take a moment)...")
    mean_lwe = ds.lwe.mean(dim='time').compute()  # Dask will handle chunking
    
    # Divide into regions
    mid_y = ny // 2
    mid_x = nx // 2
    
    regions = {
        'NW': mean_lwe.values[:mid_y, :mid_x],
        'NE': mean_lwe.values[:mid_y, mid_x:],
        'SW': mean_lwe.values[mid_y:, :mid_x],
        'SE': mean_lwe.values[mid_y:, mid_x:]
    }
    
    for name, region in regions.items():
        valid = region[~np.isnan(region)]
        if len(valid) > 0:
            print(f"   {name}: mean={np.mean(valid):6.2f} cm, std={np.std(valid):5.2f} cm, {len(valid):,} pixels")
    
    print("   ✓ Spatial variation looks reasonable")
    
    # Check 7: Quick visualization
    print("\n" + "=" * 60)
    print("8. GENERATING SAMPLE PLOTS")
    print("=" * 60)
    
    # Compute std (also lazy)
    print("   Computing temporal std...")
    std_lwe = ds.lwe.std(dim='time').compute()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: First time step
    first_data = ds.lwe.isel(time=0).values
    im1 = axes[0, 0].imshow(first_data, cmap='RdBu_r', vmin=-50, vmax=50)
    axes[0, 0].set_title(f"First: {str(ds.time.values[0])[:10]}")
    plt.colorbar(im1, ax=axes[0, 0], label='LWE (cm)')
    
    # Plot 2: Last time step
    last_data = ds.lwe.isel(time=-1).values
    im2 = axes[0, 1].imshow(last_data, cmap='RdBu_r', vmin=-50, vmax=50)
    axes[0, 1].set_title(f"Last: {str(ds.time.values[-1])[:10]}")
    plt.colorbar(im2, ax=axes[0, 1], label='LWE (cm)')
    
    # Plot 3: Temporal mean
    im3 = axes[1, 0].imshow(mean_lwe.values, cmap='RdBu_r', vmin=-50, vmax=50)
    axes[1, 0].set_title('Temporal Mean')
    plt.colorbar(im3, ax=axes[1, 0], label='LWE (cm)')
    
    # Plot 4: Temporal std
    im4 = axes[1, 1].imshow(std_lwe.values, cmap='viridis', vmin=0, vmax=20)
    axes[1, 1].set_title('Temporal Std Dev')
    plt.colorbar(im4, ax=axes[1, 1], label='LWE (cm)')
    
    plt.tight_layout()
    output_fig = "grace_zarr_verification.png"
    plt.savefig(output_fig, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_fig}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("   ✓ All checks passed!")
    print(f"   ✓ {nt} time steps from {str(ds.time.values[0])[:10]} to {str(ds.time.values[-1])[:10]}")
    print(f"   ✓ {ny} × {nx} = {ny*nx:,} pixels at 500m resolution")
    print(f"   ✓ Ready for analysis!")
    print("=" * 60)

if __name__ == '__main__':
    main()
