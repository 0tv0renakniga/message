"""
verify_step_02_all.py

-------------------------------------------------------------------------------
COMPUTATIONAL GLACIOLOGY AUDIT LOG
-------------------------------------------------------------------------------
DATE: 2026-02-07
AUTHOR: SYSTEM (Skeptical Gatekeeper)
STATUS: VERIFICATION (GENERALIZED)
LOGIC:  Iterative audit of Height Change (delta_h) and Rates (dhdt).
        Handles variable name changes and unit labeling dynamically.
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
DASK_CONFIG = {
    "n_workers": 2,
    "threads_per_worker": 2,
    "memory_limit": "8GB"
}

# Define the products to audit
PRODUCTS = {
    'delta_h': {
        'path': "data/processed_layers/intermediate/icesat2_1km_seamless_deltah.zarr",
        'main_var': 'delta_h',
        'sigma_var': 'delta_h_sigma',
        'units': 'm',
        'title': 'Cumulative Height Change'
    },
    'lag1': {
        'path': "data/processed_layers/intermediate/icesat2_1km_seamless_lag1.zarr",
        'main_var': 'dhdt',
        'sigma_var': 'dhdt_sigma',
        'units': 'm/yr',
        'title': 'Quarterly Elevation Change Rate (Lag 1)'
    },
    'lag4': {
        'path': "data/processed_layers/intermediate/icesat2_1km_seamless_lag4.zarr",
        'main_var': 'dhdt',
        'sigma_var': 'dhdt_sigma',
        'units': 'm/yr',
        'title': 'Annual Elevation Change Rate (Lag 4)'
    }
}

def verify_zarr(ds: xr.Dataset, meta: dict):
    """
    Conducts a statistical audit of the dataset.
    """
    key = meta['main_var']
    print(f"\n[Audit] Checking Data Integrity for '{key}'...")
    
    # 1. Check Dimensions
    print(f"  > Dimensions: {dict(ds.sizes)}")
    if 'x' not in ds.coords or 'y' not in ds.coords:
        raise ValueError("Missing spatial coordinates (x, y).")

    # 2. Check for Content (Middle Timestep)
    mid_idx = ds.sizes['time'] // 2
    sample = ds[key].isel(time=mid_idx)
    
    print("  > Computing sample statistics...")
    valid_mask = sample.notnull()
    valid_count = valid_mask.sum().compute().item()
    total_count = sample.size
    
    print(f"  > Valid Pixels: {valid_count} / {total_count} ({valid_count/total_count:.1%})")
    
    if valid_count == 0:
        raise ValueError(f"[FATAL] Zarr variable '{key}' contains NO valid data.")
    
    # 3. Physical Range Check
    vmin = sample.min().compute().item()
    vmax = sample.max().compute().item()
    print(f"  > Physical Range: {vmin:.2f} {meta['units']} to {vmax:.2f} {meta['units']}")
    
    limit = 500
    if abs(vmin) > limit or abs(vmax) > limit:
         print(f"  [WARN] Extreme values detected (> +/- {limit}).")
    else:
         print(f"  [PASS] Values within glaciological limits (+/- {limit}).")


def plot_diagnostics(ds: xr.Dataset, meta: dict, product_name: str):
    """
    Generates a 4-panel diagnostic plot dynamically.
    """
    print("\n[Plot] Generating Diagnostic Visuals...")
    
    var_name = meta['main_var']
    sigma_name = meta['sigma_var']
    units = meta['units']
    
    # Select Last Timestep for cumulative/latest state
    ds_last = ds.isel(time=-1).compute()
    
    # Flatten Axes
    fig, axes = plt.subplots(2, 2, figsize=(18, 16), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    # --- Panel 1: Main Variable ---
    print(f"  > Plotting {var_name} map...")
    ds_last[var_name].plot(ax=ax1, cmap='RdBu_r', robust=True, cbar_kwargs={'label': f'{var_name} ({units})'})
    ax1.set_title(f"{meta['title']} (Last Step: {np.datetime_as_string(ds.time[-1].values, unit='D')})")
    ax1.set_aspect('equal')
    
    # --- Panel 2: Uncertainty ---
    print(f"  > Plotting {sigma_name} map...")
    if sigma_name in ds_last:
        ds_last[sigma_name].plot(ax=ax2, cmap='viridis', vmax=0.5, cbar_kwargs={'label': 'Sigma (m)'})
        ax2.set_title("Uncertainty Field (Check for Seams/Patchwork)")
    else:
        ax2.text(0.5, 0.5, f"{sigma_name} Missing", ha='center')
    ax2.set_aspect('equal')
    
    # --- Panel 3: Histogram ---
    print("  > Plotting Histogram...")
    data_flat = ds_last[var_name].values.flatten()
    data_flat = data_flat[~np.isnan(data_flat)]
    
    # Dynamic Range based on units
    # m (delta_h) -> +/- 20, m/yr (dhdt) -> +/- 10
    rng = (-20, 20) if units == 'm' else (-5, 5)
    
    ax3.hist(data_flat, bins=100, range=rng, color='steelblue', edgecolor='black', alpha=0.7)
    ax3.set_title(f"Distribution ({rng} to {rng[1]} {units})")
    ax3.set_xlabel(f"{var_name} ({units})")
    ax3.set_yscale('log')
    ax3.grid(True, which="both", alpha=0.3)
    
    # --- Panel 4: Time Series ---
    print("  > Plotting Time Series...")
    if 'ice_area' in ds:
        weights = ds['ice_area'].fillna(0)
        ts = ds[var_name].weighted(weights).mean(dim=['x', 'y']).compute()
    else:
        ts = ds[var_name].mean(dim=['x', 'y']).compute()
        
    ax4.plot(ts.time, ts.values, marker='o', linestyle='-', color='darkred')
    ax4.set_title(f"Continental Mean {meta['title']}")
    ax4.set_ylabel(units)
    ax4.grid(True)
    
    # Save
    out_file = f"verify_step_02_{product_name}.png"
    plt.savefig(out_file, dpi=150)
    print(f"\n[Success] Plot saved: {os.path.abspath(out_file)}")


def main():
    cluster = LocalCluster(**DASK_CONFIG)
    client = Client(cluster)
    print(f"[System] Dask Client: {client.dashboard_link}")
    
    for name, meta in PRODUCTS.items():
        print(f"\n{'='*60}")
        print(f"PROCESSING PRODUCT: {name.upper()}")
        print(f"{'='*60}")
        
        path = meta['path']
        if not os.path.exists(path):
            print(f"[Skip] File not found: {path}")
            continue
            
        try:
            # Open Zarr
            ds = xr.open_zarr(path, consolidated=False)
            
            # Verify & Plot
            verify_zarr(ds, meta)
            plot_diagnostics(ds, meta, name)
            
        except Exception as e:
            print(f"[ERROR] Failed to verify {name}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n[System] All Verifications Complete.")

if __name__ == "__main__":
    main()
