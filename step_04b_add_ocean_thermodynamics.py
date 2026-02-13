"""
step_04b_add_ocean_thermodynamics.py

-------------------------------------------------------------------------------
COMPUTATIONAL GLACIOLOGY - OPTIMISED PIPELINE
-------------------------------------------------------------------------------
DATE:   2026-02-13
STATUS: PRODUCTION
LOGIC:
  Adds freezing point T_f and thermal driving T* to the matched ocean grid.

  Physics
  -------
  T_f(S, P) = λ₁·S + λ₂ + λ₃·P        Millero (1978) / Jenkins (1991)
  T*        = θ - T_f                    Holland & Jenkins (1999)

  where
    S  = practical salinity [PSU]             from GLORYS at clamped draft
    θ  = potential temperature [°C]           from GLORYS at clamped draft
    P  = max(0, -ice_draft) [dbar ≈ m]       depth of ice base below sea level

  CRITICAL FIX
  ------------
  The original code used `ice_draft` directly as pressure.  Because
  ice_draft is NEGATIVE for sub-sea-level ice bases (surface - thickness),
  this REVERSED the pressure-dependent freezing-point depression:

      Original:  T_f(P = -2483)  →  T_f too warm by +3.78 °C
      Fixed:     T_f(P = +2483)  →  correct freezing-point depression

  The error systematically overestimated T_f and underestimated T*,
  which would underestimate basal melt rates everywhere.

  PERFORMANCE
  -----------
  • Filesystem copy of the input store (~12 GB) avoids decompressing and
    recompressing the 6 existing variables through Dask.
  • Only T_f and T_star are computed and written (mode="a"), not the full
    8-variable dataset.
  • No Dask LocalCluster — the computation is trivially element-wise.
    The default threaded scheduler handles chunked reads/writes.
  • numcodecs.Blosc for zarr-python ≥ 3.0 forward compatibility.
-------------------------------------------------------------------------------
"""

import os
import shutil
import time as _time
import warnings

import numpy as np
import xarray as xr
import zarr
import numcodecs

warnings.filterwarnings("ignore")
xr.set_options(keep_attrs=True)

# ── Configuration ───────────────────────────────────────────────────────────
INPUT_ZARR  = "data/processed_layers/matched_ocean_grid.zarr"
OUTPUT_ZARR = "data/processed_layers/thermodynamic_ocean_grid.zarr"
CHUNK_SIZE  = 2048

# Millero (1978) / Jenkins (1991) freezing-point coefficients
#   T_f = λ₁·S + λ₂ + λ₃·P
LAMBDA_1 = -0.0575     # °C / PSU
LAMBDA_2 =  0.0901     # °C
LAMBDA_3 = -7.61e-4    # °C / dbar  (P positive downward)


# ── Pipeline ────────────────────────────────────────────────────────────────

def run_pipeline():
    wall = _time.perf_counter()
    print("=" * 72)
    print(" Step 04b · Ocean Thermodynamics (T_f, T*)")
    print("=" * 72)

    # ── 1. Copy input store (preserves existing variables as-is) ────────
    #
    # shutil.copytree duplicates the compressed Zarr chunks at the
    # filesystem level — no Dask, no decompression, no recompression.
    # For ~12 GB of zstd-compressed chunks this takes seconds, vs.
    # minutes for a full read-compute-write of all 6 original variables.
    #
    if os.path.exists(OUTPUT_ZARR):
        shutil.rmtree(OUTPUT_ZARR)

    print(f"[Copy]   {INPUT_ZARR}")
    print(f"      →  {OUTPUT_ZARR}")
    t0 = _time.perf_counter()
    shutil.copytree(INPUT_ZARR, OUTPUT_ZARR)
    print(f"[Copy]   Done  [{_time.perf_counter() - t0:.1f} s]")

    # ── 2. Remove stale consolidated metadata ───────────────────────────
    #
    # The copied .zmetadata lists only the original 6 variables.
    # Remove it so that open_zarr reads per-array metadata instead.
    # We re-consolidate at the end after appending T_f and T_star.
    #
    zmetadata_path = os.path.join(OUTPUT_ZARR, ".zmetadata")
    if os.path.exists(zmetadata_path):
        os.remove(zmetadata_path)

    # ── 3. Open the copy (Dask-backed, chunk-aligned) ──────────────────
    ds = xr.open_zarr(
        OUTPUT_ZARR,
        consolidated=False,
        chunks={"time": 1, "y": CHUNK_SIZE, "x": CHUNK_SIZE},
    )

    nT = len(ds.time)
    ny, nx = len(ds.y), len(ds.x)
    print(f"[Grid]   {ny}×{nx}  |  {nT} time steps")

    # ── 4. Pressure at the ice-ocean interface ─────────────────────────
    #
    # ice_draft = surface - thickness [m].
    # Negative when the ice base is below sea level (the normal case).
    # Pressure in dbar ≈ depth below sea level in metres:
    #
    #   P = max(0, -ice_draft)
    #
    # Pixels where ice_draft is NaN → pressure is NaN → T_f is NaN.
    # Pixels where ice_draft > 0 (base above sea level) → P = 0.
    #
    pressure_dbar = (-ds.ice_draft).clip(min=0)    # (y, x), ≥ 0

    p_max = float(pressure_dbar.max().compute())
    print(f"[Phys]   Pressure range: 0 → {p_max:.0f} dbar")

    # ── 5. Freezing point ──────────────────────────────────────────────
    #
    # T_f = λ₁·S + λ₂ + λ₃·P
    #
    # Computation is float64 for precision, then cast to float32 for
    # storage.  Broadcasting: so (time, y, x) × pressure_dbar (y, x)
    # → T_f (time, y, x).
    #
    print("[Phys]   Computing T_f = λ₁·S + λ₂ + λ₃·P …")
    T_f = (
        LAMBDA_1 * ds.so + LAMBDA_2 + LAMBDA_3 * pressure_dbar
    ).astype(np.float32)

    T_f.name = "T_f"
    T_f.attrs = {
        "long_name": "In-situ freezing point at ice-ocean interface",
        "units": "degrees_C",
        "formula": "T_f = lambda_1 * S + lambda_2 + lambda_3 * P",
        "lambda_1": LAMBDA_1,
        "lambda_2": LAMBDA_2,
        "lambda_3": LAMBDA_3,
        "pressure_source": "max(0, -ice_draft)  [dbar ~ m seawater]",
        "reference": "Millero (1978); Jenkins (1991)",
    }

    # ── 6. Thermal driving ─────────────────────────────────────────────
    #
    # T* = θ - T_f
    #
    # Positive T* means the ocean is warmer than local freezing →
    # melt potential.  Negative T* → supercooled / refreezing.
    #
    print("[Phys]   Computing T* = θ - T_f …")
    T_star = (ds.thetao - T_f).astype(np.float32)

    T_star.name = "T_star"
    T_star.attrs = {
        "long_name": "Thermal driving at ice-ocean interface",
        "units": "degrees_C",
        "description": (
            "Ocean temperature minus local freezing point.  "
            "Positive → melt potential; negative → supercooled."
        ),
    }

    # ── 7. Append new variables to the copied store ────────────────────
    ds_new = xr.Dataset({"T_f": T_f, "T_star": T_star})
    ds_new = ds_new.drop_vars("spatial_ref", errors="ignore")

    comp = numcodecs.Blosc(
        cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE
    )
    enc = {
        "T_f":    {"compressor": comp, "chunks": (1, CHUNK_SIZE, CHUNK_SIZE)},
        "T_star": {"compressor": comp, "chunks": (1, CHUNK_SIZE, CHUNK_SIZE)},
    }

    print("[Write]  Appending T_f and T_star to store …")
    t0 = _time.perf_counter()
    ds_new.to_zarr(OUTPUT_ZARR, mode="a", encoding=enc)
    print(f"[Write]  Done  [{_time.perf_counter() - t0:.1f} s]")

    # ── 8. Re-consolidate metadata ─────────────────────────────────────
    zarr.consolidate_metadata(OUTPUT_ZARR)

    # ── 9. Quick sanity check ──────────────────────────────────────────
    ds_out = xr.open_zarr(OUTPUT_ZARR)
    n_vars = len(ds_out.data_vars)
    tf_min = float(ds_out.T_f.min().compute())
    tf_max = float(ds_out.T_f.max().compute())
    ts_min = float(ds_out.T_star.min().compute())
    ts_max = float(ds_out.T_star.max().compute())

    print(f"\n[Check]  Output variables: {n_vars}  "
          f"({', '.join(sorted(ds_out.data_vars))})")
    print(f"[Check]  T_f   range: [{tf_min:.3f}, {tf_max:.3f}] °C")
    print(f"[Check]  T*    range: [{ts_min:.3f}, {ts_max:.3f}] °C")

    wall_t = _time.perf_counter() - wall
    print(f"\n{'=' * 72}")
    print(f" Done.  {wall_t:.0f} s ({wall_t / 60:.1f} min)")
    print(f"{'=' * 72}")


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()
