"""
compute_spatial_features.py

-------------------------------------------------------------------------------
COMPUTATIONAL GLACIOLOGY - OPTIMISED PIPELINE
-------------------------------------------------------------------------------
DATE:   2026-02-13
STATUS: PRODUCTION

Computes engineered spatial features on the Bedmap3 500 m grid:

  h_surface_dynamic        bedmap.surface + icesat2.delta_h
  bed_slope                |nabla(bed)|                        [m/m]
  surface_slope            |nabla(h_surface_dynamic)|          [m/m]
  dist_to_grounding_line   EDT from grounded-ice pixels        [m]

All outputs are masked to ice (grounded=1 + floating shelf=3).

Fixes
-----
1. Gradient uses dask.array.map_overlap (depth=1, boundary='nearest')
   instead of apply_ufunc(dask='parallelized').  This:
     - adds a 1-pixel halo from the neighbouring chunk before calling
       np.gradient, then trims → correct central differences everywhere.
     - guarantees every padded chunk has >= 3 elements along spatial axes,
       eliminating the 'array too small to calculate gradient' crash.
2. Both stores opened with explicit 2048x2048 spatial chunks so Dask
   never creates millions of rechunk tasks from misaligned native chunks
   (was generating a 1.13 GiB task graph).
3. No Dask distributed cluster - the workload is element-wise arithmetic
   plus one global EDT.  The default threaded scheduler is sufficient.
4. numcodecs.Blosc for zarr-python v3 forward compatibility.
-------------------------------------------------------------------------------
"""

import os
import shutil
import time as _time
import warnings

import numpy as np
import xarray as xr
import dask.array as da
import numcodecs
import zarr
from scipy.ndimage import distance_transform_edt

warnings.filterwarnings("ignore")
xr.set_options(keep_attrs=True)

# ── Configuration ───────────────────────────────────────────────────────────
BEDMAP_ZARR = "data/processed/bedmap3_500m.zarr"
ICESAT_ZARR = "data/processed/icesat2_500m_deltah.zarr"
OUTPUT_ZARR = "data/processed/spatial_features_engineered.zarr"
CHUNK_YX    = 2048
DX = DY     = 500.0      # Bedmap3 grid spacing [m]


# ── Slope magnitude ────────────────────────────────────────────────────────

def compute_slope(da_arr, dy=DY, dx=DX):
    """
    Compute |∇f| via dask.array.map_overlap.

    A 1-pixel halo is fetched from each neighbouring chunk (or padded
    with 'nearest' at the array boundary) before calling np.gradient.
    After trimming, every output pixel — including those at chunk
    boundaries — has a correct central-difference gradient.

    This also prevents the 'array too small' crash: a chunk of even
    1 element is padded to 3, which satisfies np.gradient's minimum
    requirement of 2 elements.
    """
    y_ax = da_arr.dims.index("y")
    x_ax = da_arr.dims.index("x")

    def _grad_magnitude(block):
        gy, gx = np.gradient(block, dy, dx, axis=(y_ax, x_ax))
        return np.sqrt(gx**2 + gy**2).astype(np.float32)

    overlap = {i: 1 if i in (y_ax, x_ax) else 0
               for i in range(da_arr.ndim)}

    result = da.map_overlap(
        _grad_magnitude,
        da_arr.data,
        depth=overlap,
        boundary="nearest",
        dtype=np.float32,
    )
    return xr.DataArray(result, coords=da_arr.coords, dims=da_arr.dims)


# ── Grounding-line distance ────────────────────────────────────────────────

def compute_gl_distance(mask_da, dx=DX):
    """
    Euclidean distance to the nearest grounded-ice pixel (mask == 1).

    The EDT is a global operation — the full 12288x12288 mask must be
    in memory (~150 MB as uint8).  The result is wrapped back into a
    chunked Dask array for lazy downstream writes.
    """
    print("[Feat]   dist_to_grounding_line (EDT, eager)")
    mask_2d = mask_da.values
    grounded = (mask_2d == 1)
    dist = distance_transform_edt(~grounded, sampling=dx).astype(np.float32)

    return xr.DataArray(
        da.from_array(dist, chunks=(CHUNK_YX, CHUNK_YX)),
        coords=mask_da.coords,
        dims=mask_da.dims,
        attrs={"long_name": "Distance to grounding line", "units": "m"},
    )


# ── Pipeline ───────────────────────────────────────────────────────────────

def main():
    wall = _time.perf_counter()
    print("=" * 72)
    print(" Compute Spatial Features")
    print("=" * 72)

    # ── 1. Open stores with identical 2048x2048 spatial chunks ──────────
    #
    # Forcing the same spatial chunk layout on both stores prevents
    # Dask from creating a huge rechunk graph when the native Zarr
    # chunk sizes differ between bedmap and icesat2.
    #
    chunks_yx = {"y": CHUNK_YX, "x": CHUNK_YX}

    bedmap    = xr.open_zarr(BEDMAP_ZARR,  chunks=chunks_yx)
    icesat_dh = xr.open_zarr(ICESAT_ZARR,  chunks=chunks_yx)

    master_mask = bedmap["mask"]
    ny, nx = len(bedmap.y), len(bedmap.x)
    print(f"[Grid]   {ny}x{nx}")

    # ── 2. Dynamic surface elevation ────────────────────────────────────
    print("[Feat]   h_surface_dynamic = surface + delta_h")
    h_surface = (bedmap["surface"] + icesat_dh["delta_h"]).astype(np.float32)

    # ── 3. Slope magnitudes (map_overlap) ───────────────────────────────
    print("[Feat]   bed_slope = |∇(bed)|")
    bed_slope = compute_slope(bedmap["bed"])
    bed_slope.attrs = {"long_name": "Bed topography slope", "units": "m/m"}

    print("[Feat]   surface_slope = |∇(h_surface)|")
    surface_slope = compute_slope(h_surface)
    surface_slope.attrs = {"long_name": "Dynamic surface slope", "units": "m/m"}

    # ── 4. Grounding-line distance ──────────────────────────────────────
    gl_distance = compute_gl_distance(master_mask)

    # ── 5. Assemble dataset ─────────────────────────────────────────────
    ds_features = xr.Dataset({
        "h_surface_dynamic":      h_surface,
        "bed_slope":              bed_slope,
        "surface_slope":          surface_slope,
        "dist_to_grounding_line": gl_distance,
    })

    # ── 6. Mask to ice only (grounded=1, floating shelf=3) ─────────────
    print("[Mask]   Keeping grounded ice (1) + floating shelf (3)")
    valid_ice = (master_mask == 1) | (master_mask == 3)
    ds_features = ds_features.where(valid_ice, drop=False)

    # ── 7. Write to Zarr ────────────────────────────────────────────────
    if os.path.exists(OUTPUT_ZARR):
        shutil.rmtree(OUTPUT_ZARR)

    comp = numcodecs.Blosc(
        cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE
    )
    encoding = {}
    for var in ds_features.data_vars:
        dims = ds_features[var].dims
        chunks = tuple(1 if d == "time" else CHUNK_YX for d in dims)
        encoding[var] = {"compressor": comp, "chunks": chunks}

    print(f"[Write]  → {OUTPUT_ZARR}")
    t0 = _time.perf_counter()
    ds_features.to_zarr(
        OUTPUT_ZARR, mode="w", encoding=encoding,
        zarr_format=2, consolidated=False,
    )
    zarr.consolidate_metadata(OUTPUT_ZARR)
    print(f"[Write]  Done  [{_time.perf_counter() - t0:.1f} s]")

    wall_t = _time.perf_counter() - wall
    print(f"\n{'=' * 72}")
    print(f" Done.  {wall_t:.0f} s ({wall_t / 60:.1f} min)")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
