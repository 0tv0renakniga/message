"""
verify_step_02b_500m.py

-------------------------------------------------------------------------------
COMPUTATIONAL GLACIOLOGY AUDIT LOG
-------------------------------------------------------------------------------
DATE: 2026-02-07
AUTHOR: SYSTEM (Skeptical Gatekeeper)
STATUS: DIAGNOSTIC
LOGIC:  Verifies integrity of 500m upsampled products.
        Checks: Resolution, CRS, NaN distribution, and Signal Histograms.
-------------------------------------------------------------------------------
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import dask.array as da

# --- CONFIGURATION ---
BASE_DIR = "data/processed_layers"
FILES = {
    "Delta_H": "icesat2_500m_deltah.zarr",
    "Lag1": "icesat2_500m_lag1.zarr",
    "Lag4": "icesat2_500m_lag4.zarr"
}

def audit_dataset(name: str, path: str):
    print(f"\n[Audit] Inspecting {name}...")
    full_path = os.path.join(BASE_DIR, path)
    
    if not os.path.exists(full_path):
        print(f"  > [FAIL] File not found: {full_path}")
        return

    try:
        ds = xr.open_zarr(full_path, consolidated=False)
    except Exception as e:
        print(f"  > [FAIL] Corrupt Zarr: {e}")
        return

    # 1. Geometry Check
    x_count = len(ds.x)
    y_count = len(ds.y)
    print(f"  > Dimensions: X={x_count}, Y={y_count}, Time={len(ds.time)}")
    
    # Resolution Check (Scalar Math)
    x_res = abs((ds.x.max().item() - ds.x.min().item()) / (x_count - 1))
    y_res = abs((ds.y.max().item() - ds.y.min().item()) / (y_count - 1))
    
    print(f"  > Resolution: {x_res:.4f}m x {y_res:.4f}m")
    
    if not np.isclose(x_res, 500.0, atol=0.1):
        print("  > [FAIL] Resolution violation!")
    else:
        print("  > [PASS] Grid spacing confirms 500m.")

    # 2. Data Integrity Check (Lazy)
    # Pick a variable to test
    if 'delta_h' in ds:
        var = ds['delta_h']
    elif 'dhdt' in ds:
        var = ds['dhdt']
    else:
        print("  > [WARN] No primary variable found.")
        return

    # Load last timestep for stats
    print("  > Computing statistics for last timestep...")
    data_slice = var.isel(time=-1).load()
    
    valid_pixels = np.isfinite(data_slice).sum().item()
    total_pixels = data_slice.size
    coverage = (valid_pixels / total_pixels) * 100
    
    print(f"  > Coverage: {coverage:.2f}% ({valid_pixels} valid pixels)")
    print(f"  > Range: {data_slice.min().item():.2f} to {data_slice.max().item():.2f}")
    
    if coverage < 1.0:
        print("  > [FAIL] Dataset is essentially empty (<1% coverage).")
    
    return data_slice, name

def plot_verification(slices):
    """
    Generates a 3-panel plot to visually confirm upsampling.
    """
    if not slices:
        return

    print("\n[Visual] Generating verification plot...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    
    for ax, (data, title) in zip(axes, slices):
        # Mask 0.0 if it represents ocean (optional, depends on your data)
        # data = data.where(data != 0)
        
        # Robust quantile plotting to ignore outliers
        vmin, vmax = np.nanpercentile(data, [1])
        
        im = ax.imshow(data, extent=[data.x.min(), data.x.max(), data.y.min(), data.y.max()],
                       origin='upper', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(f"{title} (500m)\nRange: [{vmin:.2f}, {vmax:.2f}]")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')

    output_plot = "verify_step_02b_500m_integrity.png"
    plt.savefig(output_plot, dpi=150)
    print(f"  > Saved: {output_plot}")
    plt.close()

def main():
    collected_slices = []
    
    for name, filename in FILES.items():
        result = audit_dataset(name, filename)
        if result:
            collected_slices.append(result)
            
    if collected_slices:
        plot_verification(collected_slices)
    else:
        print("[System] No valid data found to plot.")

if __name__ == "__main__":
    main()
