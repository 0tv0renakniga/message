"""
verify_step_02a.py

-------------------------------------------------------------------------------
COMPUTATIONAL GLACIOLOGY AUDIT LOG
-------------------------------------------------------------------------------
DATE: 2026-02-07
AUTHOR: SYSTEM (Skeptical Gatekeeper)
STATUS: VERIFICATION
LOGIC:  Audit intermediate Zarr for physical realism and seam artifacts.
-------------------------------------------------------------------------------
"""

import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import dask.array as da
from dask.distributed import Client, LocalCluster

# --- CONFIGURATION ---
# Use a lightweight LocalCluster for verification (don't need 55GB for this)
DASK_CONFIG = {
    "n_workers": 2,
    "threads_per_worker": 2,
    "memory_limit": "8GB"
}

INPUT_ZARR = "data/processed_layers/intermediate/icesat2_1km_seamless_deltah.zarr"

def verify_zarr(ds: xr.Dataset):
    """
    Conducts a statistical audit of the dataset.
    """
    print("\n[Audit] Checking Data Integrity...")
    
    # 1. Check Dimensions
    print(f"  > Dimensions: {dict(ds.sizes)}")
    if 'x' not in ds.coords or 'y' not in ds.coords:
        raise ValueError("Missing spatial coordinates (x, y).")

    # 2. Check for Content (Load a sample)
    # We pick the MIDDLE timestamp to avoid edge effects of start/end mission
    mid_idx = ds.sizes['time'] // 2
    sample = ds['delta_h'].isel(time=mid_idx)
    
    print("  > Computing sample statistics (this may take 30s)...")
    valid_mask = sample.notnull()
    valid_count = valid_mask.sum().compute().item()
    total_count = sample.size
    
    print(f"  > Valid Pixels: {valid_count} / {total_count} ({valid_count/total_count:.1%})")
    
    if valid_count == 0:
        raise ValueError("[FATAL] Zarr contains NO valid data (All NaNs).")
    
    # 3. Physical Range Check
    vmin = sample.min().compute().item()
    vmax = sample.max().compute().item()
    print(f"  > Physical Range (Delta_H): {vmin:.2f} m to {vmax:.2f} m")
    
    if abs(vmin) > 500 or abs(vmax) > 500:
         print("  [WARN] Extreme values detected (possible outliers).")
    else:
         print("  [PASS] Values within glaciological limits (+/- 500m).")


def plot_diagnostics(ds: xr.Dataset):
    """
    Generates a 4-panel diagnostic plot.
    """
    print("\n[Plot] Generating Diagnostic Visuals...")
    
    # Select Last Timestep (Cumulative Change is most visible here)
    ds_last = ds.isel(time=-1).compute()
    
    # Create Figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 16), constrained_layout=True)
    
    # Panel 1: Delta_H (Height Change)
    # Robust=True trims outliers for better contrast
    ds_last['delta_h'].plot(ax=axes, cmap='RdBu_r', robust=True, cbar_kwargs={'label': 'Delta_H (m)'})
    axes.set_title(f"Height Change (Last Timestep: {np.datetime_as_string(ds.time[-1].values, unit='D')})")
    axes.set_aspect('equal')
    
    # Panel 2: Delta_H_Sigma (Uncertainty) - THE SEAM CHECKER
    # We expect to see a 'patchwork' look if tiles were merged correctly.
    # If it's uniform, the uncertainty merge failed.
    if 'delta_h_sigma' in ds_last:
        ds_last['delta_h_sigma'].plot(ax=axes[1], cmap='viridis', vmax=0.5, cbar_kwargs={'label': 'Sigma (m)'})
        axes[1].set_title("Uncertainty Field (Check for Seams/Patchwork)")
        axes[1].set_aspect('equal')
    
    # Panel 3: Histogram of Values
    data_flat = ds_last['delta_h'].values.flatten()
    data_flat = data_flat[~np.isnan(data_flat)]
    axes[1].hist(data_flat, bins=100, range=(-20, 20), color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].set_title("Distribution of Height Change (+/- 20m)")
    axes[1].set_xlabel("Meters")
    axes[1].set_yscale('log') # Log scale to see outliers
    
    # Panel 4: Time Series (Continental Mean)
    print("  > Computing Time Series (this calculates the mean of the whole array)...")
    # We weight by ice_area if available, otherwise raw mean
    if 'ice_area' in ds:
        weights = ds['ice_area'].fillna(0)
        ts = ds['delta_h'].weighted(weights).mean(dim=['x', 'y']).compute()
    else:
        ts = ds['delta_h'].mean(dim=['x', 'y']).compute()
        
    axes[1].plot(ts.time, ts.values, marker='o', linestyle='-', color='darkred')
    axes[1].set_title("Continental Mean Height Change Trend")
    axes[1].grid(True)
    
    # Save
    out_file = "verify_step_02a_deltah.png"
    plt.savefig(out_file, dpi=150)
    print(f"\n[Success] Diagnostic plot saved to: {os.path.abspath(out_file)}")
    print("  > Inspect the 'Uncertainty Field' (Top Right).")
    print("  > It should look like a mosaic (sharp transitions are GOOD here, it means we selected the best sensor).")


def main():
    # 1. Start Client
    cluster = LocalCluster(**DASK_CONFIG)
    client = Client(cluster)
    print(f"[System] Dask Client: {client.dashboard_link}")
    
    # 2. Open Zarr
    if not os.path.exists(INPUT_ZARR):
        print(f"[FATAL] Zarr not found: {INPUT_ZARR}")
        sys.exit(1)
        
    ds = xr.open_zarr(INPUT_ZARR, consolidated=False) # Consolidated might not be written yet
    
    # 3. Verify
    try:
        verify_zarr(ds)
        plot_diagnostics(ds)
    except Exception as e:
        print(f"\n[FATAL] Verification Failed: {e}")
        # traceback.print_exc()
    
    print("[System] Verification Complete.")

if __name__ == "__main__":
    main()
