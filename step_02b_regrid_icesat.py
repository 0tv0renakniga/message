"""
step_02b_regrid_icesat.py

-------------------------------------------------------------------------------
COMPUTATIONAL GLACIOLOGY AUDIT LOG
-------------------------------------------------------------------------------
DATE: 2026-02-07
AUTHOR: SYSTEM (Skeptical Gatekeeper)
STATUS: PRODUCTION (Architecture Correction)
LOGIC:  Upsample 1km ICESat-2 to 500m Master Grid using Native Interpolation.
        REMOVED xeSMF (Spherical) to prevent Memory Crash on Cartesian grids.
-------------------------------------------------------------------------------
"""

import os
import sys
import warnings
import numpy as np
import xarray as xr
import dask.config
from dask.distributed import Client, LocalCluster

# --- MANDATORY PREAMBLE ---
warnings.filterwarnings("ignore") 
xr.set_options(keep_attrs=True)

# --- CONFIGURATION ---
# We can now increase workers because memory pressure is 10x lower.
DASK_CONFIG = {
    "n_workers": 6, 
    "threads_per_worker": 1, 
    "memory_limit": "8GB"
}

# Memory Tuning
dask.config.set({
    "distributed.worker.memory.target": 0.70,
    "distributed.worker.memory.spill": 0.85,
    "distributed.worker.memory.pause": 0.95,
})

DIRS = {
    "input": "data/processed_layers/intermediate",
    "output": "data/processed_layers",
    "master": "data/processed_layers/master_grid_template.nc"
}

TASKS = {
    'delta_h': {
        'input': 'icesat2_1km_seamless_deltah.zarr',
        'output': 'icesat2_500m_deltah.zarr',
        'var_list': ['delta_h', 'delta_h_sigma', 'ice_area']
    },
    'lag1': {
        'input': 'icesat2_1km_seamless_lag1.zarr',
        'output': 'icesat2_500m_lag1.zarr',
        'var_list': ['dhdt', 'dhdt_sigma', 'ice_area']
    },
    'lag4': {
        'input': 'icesat2_1km_seamless_lag4.zarr',
        'output': 'icesat2_500m_lag4.zarr',
        'var_list': ['dhdt', 'dhdt_sigma', 'ice_area']
    }
}

def verify_resolution_compatibility(src: xr.Dataset, dst: xr.Dataset):
    """
    Ensures the 1km grid and 500m grid are resolution-compatible.
    Uses robust SCALAR math to avoid the 'Scalar-Array' crash.
    """
    print("[Audit] Verifying Grid Compatibility...")
    
    # 1. Get Scalars
    src_xmin, src_xmax = src.x.min().item(), src.x.max().item()
    dst_xmin, dst_xmax = dst.x.min().item(), dst.x.max().item()
    
    # 2. Get Counts (Integers)
    src_n = len(src.x)
    dst_n = len(dst.x)
    
    # 3. Calculate Resolution (Span / Steps)
    src_res = abs((src_xmax - src_xmin) / (src_n - 1))
    dst_res = abs((dst_xmax - dst_xmin) / (dst_n - 1))
    
    print(f"  > Source Res: {src_res:.4f}m | Target Res: {dst_res:.4f}m")
    
    if not np.isclose(src_res, 1000.0, atol=1.0):
        raise ValueError(f"Source resolution is not 1km! Found {src_res}")
    
    if not np.isclose(dst_res, 500.0, atol=0.5):
        raise ValueError(f"Target resolution is not 500m! Found {dst_res}")

    print("  [PASS] Resolutions are valid.")


def process_task(task_key: str, config: dict, ds_master: xr.Dataset, client):
    """
    Executes the interpolation for a single product.
    """
    in_path = os.path.join(DIRS['input'], config['input'])
    out_path = os.path.join(DIRS['output'], config['output'])
    
    if not os.path.exists(in_path):
        print(f"[Skip] Input not found: {in_path}")
        return

    if os.path.exists(out_path):
        print(f"[Skip] Output exists: {out_path}")
        return

    print(f"\n[Task] Processing {task_key} -> {config['output']}...")
    
    # 1. Open Source (Lazy)
    ds_src = xr.open_zarr(in_path, consolidated=False)
    ds_subset = ds_src[config['var_list']]
    
    # 2. Native Interpolation
    # strictly bilinear (linear in 2D), lazy, handles chunks automatically.
    print("  > Applying Native Bilinear Interpolation (xr.interp)...")
    
    # We pass the Master Grid coordinates. 
    # method='linear' is Bilinear for 2D grids.
    # kwargs={'fill_value': np.nan} prevents extrapolation errors at edges.
    ds_regridded = ds_subset.interp(
        x=ds_master.x, 
        y=ds_master.y, 
        method="linear",
        kwargs={"fill_value": np.nan} 
    )
    
    # 3. Post-Process
    ds_regridded.attrs = ds_src.attrs
    ds_regridded.attrs['history'] = f"Upsampled to 500m using xr.interp (linear). Source: {config['input']}"
    ds_regridded.attrs['resolution'] = "500m"
    
    # 4. Cast to float32 (Crucial for Storage)
    for var in ds_regridded.data_vars:
        if ds_regridded[var].dtype == np.float64:
             ds_regridded[var] = ds_regridded[var].astype(np.float32)
        ds_regridded[var].attrs = ds_src[var].attrs
    
    # 5. Write to Zarr
    print(f"  > Writing to {out_path}...")
    
    # Chunking: 500m grid is 12288x12288. 2048 is safe.
    ds_regridded = ds_regridded.chunk({'time': 1, 'y': 2048, 'x': 2048})
    
    compressor = dict(compressor=None)
    encoding = {v: compressor for v in ds_regridded.data_vars}
    
    ds_regridded.to_zarr(out_path, mode='w', encoding=encoding, compute=True)
    print(f"  > Success: {task_key}")


def main():
    cluster = LocalCluster(**DASK_CONFIG)
    client = Client(cluster)
    print(f"[System] Dask Client: {client.dashboard_link}")
    
    # Load Master Grid
    if os.path.exists(DIRS['master']):
        print(f"[System] Loading Master Grid from {DIRS['master']}")
        ds_master = xr.open_dataset(DIRS['master'])
    else:
        print("[Warn] Master Grid missing. Generating standard template...")
        x_grid = np.arange(-3072000, 3072000 + 500, 500, dtype=np.float32)
        y_grid = np.arange(3072000, -3072000 - 500, -500, dtype=np.float32)
        ds_master = xr.Dataset(coords={'x': x_grid, 'y': y_grid})
        ds_master.attrs['crs'] = "EPSG:3031"
        
    src_ref_path = os.path.join(DIRS['input'], TASKS['delta_h']['input'])
    if not os.path.exists(src_ref_path):
        raise FileNotFoundError(f"Reference input missing: {src_ref_path}")
    
    # Verify Compatibility (Using Safe Scalar Logic)
    ds_src_ref = xr.open_zarr(src_ref_path, consolidated=False)
    verify_resolution_compatibility(ds_src_ref, ds_master)
    
    # Run Tasks
    for key, cfg in TASKS.items():
        try:
            process_task(key, cfg, ds_master, client)
        except Exception as e:
            print(f"[Error] Failed {key}: {e}")
            import traceback
            traceback.print_exc()

    print("[System] Step 02b Complete.")

if __name__ == "__main__":
    main()
