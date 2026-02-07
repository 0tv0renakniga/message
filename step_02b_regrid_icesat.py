"""
step_02b_regrid_icesat.py

-------------------------------------------------------------------------------
COMPUTATIONAL GLACIOLOGY AUDIT LOG
-------------------------------------------------------------------------------
DATE: 2026-02-07
AUTHOR: SYSTEM (Skeptical Gatekeeper)
STATUS: PRODUCTION (Step 01 Logic Applied)
LOGIC:  Upsample 1km ICESat-2 to 500m Master Grid.
        generates 2D Lat/Lon coordinates for xeSMF (Source 15/Step 01).
-------------------------------------------------------------------------------
"""

import os
import sys
import warnings
import numpy as np
import xarray as xr
import xesmf as xe
import dask.config
from dask.distributed import Client, LocalCluster
from pyproj import Transformer

# --- MANDATORY PREAMBLE ---
warnings.filterwarnings("ignore")
xr.set_options(keep_attrs=True)

# --- CONFIGURATION (THE "SQUEEZE" SETUP) ---
DASK_CONFIG = {
    "n_workers": 4,
    "threads_per_worker": 2, 
    "memory_limit": "12GB"
}

# Memory Tuning
dask.config.set({
    "distributed.worker.memory.target": 0.70,
    "distributed.worker.memory.spill": 0.85,
    "distributed.worker.memory.pause": 0.95,
    "distributed.worker.memory.terminate": 0.98,
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

WEIGHT_FILE = "weights_icesat_1km_to_500m.nc"

def get_grid_geometry(ds: xr.Dataset, name: str) -> xr.Dataset:
    """
    Generates the explicit 2D Lat/Lon grid required by xeSMF.
    Replicates Step 01 Logic (Source 15).
    """
    print(f"[Geometry] Generating 2D coordinates for {name}...")
    
    # 1. Extract 1D Vectors
    x = ds.x.values
    y = ds.y.values
    
    # 2. Create 2D Meshgrid (Heavy Operation)
    # This creates full (Y, X) matrices
    xx, yy = np.meshgrid(x, y)
    
    # 3. Transform to Lat/Lon (EPSG:3031 -> EPSG:4326)
    transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xx, yy)
    
    # 4. Wrap in xarray for xeSMF
    ds_geo = xr.Dataset(
        coords={
            'lon': (['y', 'x'], lon),
            'lat': (['y', 'x'], lat)
        }
    )
    
    print(f"  > Generated geometry shape: {ds_geo.lon.shape}")
    return ds_geo

def verify_grid_compatibility(src: xr.Dataset, dst: xr.Dataset):
    """
    Ensures resolution compatibility.
    Uses safe scalar logic.
    """
    print("[Audit] Verifying Grid Compatibility...")
    
    src_xmin, src_xmax = src.x.min().item(), src.x.max().item()
    dst_xmin, dst_xmax = dst.x.min().item(), dst.x.max().item()
    
    src_n = len(src.x)
    dst_n = len(dst.x)
    
    # Resolution = Span / (Steps - 1)
    src_res = abs((src_xmax - src_xmin) / (src_n - 1))
    dst_res = abs((dst_xmax - dst_xmin) / (dst_n - 1))
    
    print(f"  > Source Res: {src_res:.4f}m | Target Res: {dst_res:.4f}m")
    
    if not np.isclose(src_res, 1000.0, atol=1.0):
        raise ValueError(f"Source resolution is not 1km! Found {src_res}")
    
    if not np.isclose(dst_res, 500.0, atol=0.5):
        raise ValueError(f"Target resolution is not 500m! Found {dst_res}")

    print("  [PASS] Resolutions are valid.")


def create_regridder(ds_src_ref, ds_dst_ref, reuse_weights=True):
    """
    Initializes xeSMF using explicit 2D Lat/Lon grids.
    """
    if os.path.exists(WEIGHT_FILE) and reuse_weights:
        print(f"[Regrid] Loading existing weights: {WEIGHT_FILE}")
        # When reusing weights, we don't need the heavy geometry, just valid placeholders
        # But to be safe, we pass the geometry if we have it, or simple ones.
        # Ideally, we construct the object to match the weight file.
        
        # We must regenerate geometry to initialize the regridder object correctly 
        # even if reading weights from disk, unless we use a "dummy" grid.
        # For safety, we generate them.
        geo_src = get_grid_geometry(ds_src_ref, "Source (1km)")
        geo_dst = get_grid_geometry(ds_dst_ref, "Target (500m)")
        
        regridder = xe.Regridder(geo_src, geo_dst, 'bilinear', 
                                 weights=WEIGHT_FILE, reuse_weights=True)
    else:
        print(f"[Regrid] Calculating NEW weights (Bilinear 1km->500m)...")
        
        # 1. Generate Heavy Geometry
        geo_src = get_grid_geometry(ds_src_ref, "Source (1km)")
        geo_dst = get_grid_geometry(ds_dst_ref, "Target (500m)")
        
        # 2. Build Regridder
        regridder = xe.Regridder(geo_src, geo_dst, 'bilinear', 
                                 filename=WEIGHT_FILE, reuse_weights=False)
        
        # 3. Force Compute
        _ = regridder.weights 
        print(f"[Regrid] Weights saved to {WEIGHT_FILE}")
        
    return regridder


def process_task(task_key: str, config: dict, regridder, client):
    """
    Executes the regridding for a single product.
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
    
    # 1. Open Source
    ds_src = xr.open_zarr(in_path, consolidated=False)
    ds_subset = ds_src[config['var_list']]
    
    # 2. Rename Dimensions for xeSMF
    # xeSMF expects dimensions to match the geometry coordinates (y, x)
    # Our data is (time, y, x). This is compatible.
    
    # 3. Apply Regridding
    print("  > Applying bilinear interpolation...")
    ds_regridded = regridder(ds_subset)
    
    # 4. Post-Process
    ds_regridded.attrs = ds_src.attrs
    ds_regridded.attrs['history'] = f"Upsampled to 500m using xeSMF bilinear. Source: {config['input']}"
    ds_regridded.attrs['resolution'] = "500m"
    
    for var in ds_regridded.data_vars:
        if ds_regridded[var].dtype == np.float64:
             ds_regridded[var] = ds_regridded[var].astype(np.float32)
        ds_regridded[var].attrs = ds_src[var].attrs
    
    # 5. Physics Audit
    if 'delta_h' in ds_regridded:
        v_max = ds_regridded['delta_h'].max().compute().item()
        if abs(v_max) > 600: 
            warnings.warn(f"Regridding overshoot detected: {v_max}")

    # 6. Write to Zarr
    print(f"  > Writing to {out_path}...")
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
        
    ds_src_ref = xr.open_zarr(src_ref_path, consolidated=False)
    
    # Verify Resolution
    verify_grid_compatibility(ds_src_ref, ds_master)
    
    # Initialize Regridder (using Step 01 Geometry Logic)
    regridder = create_regridder(ds_src_ref, ds_master)
    
    # Run Tasks
    for key, cfg in TASKS.items():
        try:
            process_task(key, cfg, regridder, client)
        except Exception as e:
            print(f"[Error] Failed {key}: {e}")
            import traceback
            traceback.print_exc()

    print("[System] Step 02b Complete.")

if __name__ == "__main__":
    main()
