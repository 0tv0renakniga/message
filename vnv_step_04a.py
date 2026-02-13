"""
verify_04a_ocean_extraction.py

Production verification of matched_ocean_grid.zarr output from step_04a.

All 3D-array operations are Dask-backed (lazy reductions) so that peak
memory stays well under 4 GB regardless of grid size.  2D fields are
small enough to materialise directly.

Checks
------
 1. Store integrity (variables, dims, dtypes, metadata consolidation)
 2. Coordinate alignment (y/x vs Bedmap3, time vs GLORYS)
 3. Zarr fill-value audit (NaN vs 0 for unwritten chunks)
 4. Spatial coverage (output vs Bedmap3 ice-shelf mask)
 5. 2D NaN-mask consistency (all static fields share one mask)
 6. Physical range checks (dist, depth, temperature, salinity)
 7. Physical consistency (clamped_depth ≤ |ice_draft|, mascon_id int)
 8. Temporal integrity (no empty steps, coverage per ice pixel)
 9. Variable metadata (units, long_name)
10. Summary statistics table
"""

import sys
import os
import argparse
import warnings
import time as _time

import numpy as np
import xarray as xr
import dask

warnings.filterwarnings("ignore")

# ── Default paths (must match step_04a_extract_ocean.py) ────────────────────
BEDMAP_PATH = "data/processed_layers/bedmap3_500m.zarr"
OCEAN_PATH  = "data/raw/ocean/antarctic_ocean_physics_2019_2025.nc"
OUTPUT_ZARR = "data/processed_layers/matched_ocean_grid.zarr"

# ── Physical range bounds for Antarctic shelf / cavity waters ───────────────
THETA_LO, THETA_HI = -3.5, 6.0       # °C  (includes warm CDW intrusions)
SO_LO,    SO_HI    = 25.0, 38.0      # PSU
DIST_MAX_M          = 1_500_000       # 1 500 km — no ice-shelf pixel is
                                      # farther than this from the ocean

# ── Bookkeeping ─────────────────────────────────────────────────────────────
_n_pass = 0
_n_fail = 0


def _chk(label: str, ok: bool, detail: str = "") -> bool:
    """Print a pass/fail line and update counters."""
    global _n_pass, _n_fail
    _n_pass += ok
    _n_fail += (not ok)
    sym = "✓" if ok else "✗"
    line = f"  [{sym}]  {label}"
    if detail:
        line += f"  —  {detail}"
    print(line)
    return ok


def _hdr(title: str) -> None:
    print(f"\n── {title} ──")


# ── Main verification routine ───────────────────────────────────────────────

def verify(
    output_zarr: str = OUTPUT_ZARR,
    bedmap_path: str = BEDMAP_PATH,
    ocean_path:  str = OCEAN_PATH,
) -> bool:
    global _n_pass, _n_fail
    _n_pass = _n_fail = 0
    wall = _time.perf_counter()

    print("=" * 72)
    print(" Verification · matched_ocean_grid.zarr")
    print("=" * 72)

    # ================================================================
    # 1. STORE INTEGRITY
    # ================================================================
    _hdr("1. Store Integrity")

    if not os.path.exists(output_zarr):
        _chk("Store exists", False, output_zarr)
        return False
    _chk("Store exists", True)

    try:
        ds = xr.open_zarr(output_zarr)
        _chk("Opens without error", True)
    except Exception as e:
        _chk("Opens without error", False, str(e))
        return False

    # Consolidated metadata
    _chk(
        "Metadata consolidated",
        os.path.exists(os.path.join(output_zarr, ".zmetadata")),
    )

    exp_2d = {"dist_to_ocean", "ice_draft", "clamped_depth", "mascon_id"}
    exp_3d = {"thetao", "so"}
    present = set(ds.data_vars)

    _chk(
        "2D variables present",
        exp_2d <= present,
        ", ".join(sorted(exp_2d))
        if exp_2d <= present
        else f"missing: {exp_2d - present}",
    )
    _chk(
        "3D variables present",
        exp_3d <= present,
        ", ".join(sorted(exp_3d))
        if exp_3d <= present
        else f"missing: {exp_3d - present}",
    )

    for v in sorted(exp_2d & present):
        _chk(f"{v} dims=(y, x)", ds[v].dims == ("y", "x"))
    for v in sorted(exp_3d & present):
        _chk(f"{v} dims=(time, y, x)", ds[v].dims == ("time", "y", "x"))
    for v in sorted(present):
        _chk(
            f"{v} dtype=float32",
            ds[v].dtype == np.float32,
            f"got {ds[v].dtype}",
        )

    ny, nx = len(ds.y), len(ds.x)
    nT = len(ds.time)
    print(f"  Grid: {ny}×{nx}  |  {nT} time steps")

    # ================================================================
    # 2. COORDINATE ALIGNMENT
    # ================================================================
    _hdr("2. Coordinate Alignment")

    ds_b = None
    if os.path.exists(bedmap_path):
        ds_b = xr.open_zarr(bedmap_path)
        _chk(
            "y ≡ Bedmap3.y",
            np.array_equal(ds.y.values, ds_b.y.values),
            f"len {len(ds.y)} vs {len(ds_b.y)}",
        )
        _chk(
            "x ≡ Bedmap3.x",
            np.array_equal(ds.x.values, ds_b.x.values),
            f"len {len(ds.x)} vs {len(ds_b.x)}",
        )
    else:
        _chk("Bedmap3 available", False, bedmap_path)

    glorys_max_d = None
    if os.path.exists(ocean_path):
        ds_oc = xr.open_dataset(ocean_path, engine="h5netcdf")
        _chk(
            "time ≡ GLORYS.time",
            np.array_equal(ds.time.values, ds_oc.time.values),
            f"len {nT} vs {len(ds_oc.time)}",
        )
        glorys_max_d = float(
            ds_oc.sel(latitude=slice(-90, -60)).depth.values.max()
        )
        ds_oc.close()
    else:
        _chk("GLORYS available", False, ocean_path)

    # ================================================================
    # 3. FILL-VALUE AUDIT
    # ================================================================
    _hdr("3. Fill-Value Audit")

    nan_fill = None
    try:
        import zarr as _z

        zg = _z.open_group(output_zarr, mode="r")
        fv = float(zg["dist_to_ocean"].fill_value)
        nan_fill = np.isnan(fv)
        _chk(
            "fill_value = NaN",
            nan_fill,
            f"fill_value={fv}"
            + ("  ⚠ land in unwritten blocks reads as 0" if not nan_fill else ""),
        )
    except Exception as ex:
        _chk("fill_value readable", False, str(ex))

    # ================================================================
    # 4. SPATIAL COVERAGE vs BEDMAP3
    # ================================================================
    _hdr("4. Spatial Coverage")

    ice_mask = None
    ice_da = None

    # Load dist_to_ocean (2D, ~576 MB float32 — always needed)
    print("  Loading dist_to_ocean …")
    dto = ds.dist_to_ocean.compute().values

    if ds_b is not None:
        print("  Loading Bedmap3 ice-shelf mask …")
        ice_mask = (
            (ds_b.mask == 2) | (ds_b.mask == 3)
        ).compute().values
        n_ice = int(ice_mask.sum())
        ice_da = xr.DataArray(
            ice_mask,
            dims=("y", "x"),
            coords={"y": ds.y, "x": ds.x},
        )
        print(f"  Bedmap3 ice-shelf pixels: {n_ice:,}")

        has_data = ~np.isnan(dto)
        covered  = ice_mask & has_data
        n_covered = int(covered.sum())
        n_missing = n_ice - n_covered
        pct = 100 * n_covered / n_ice if n_ice else 0
        _chk(
            "Ice-shelf coverage",
            n_missing == 0,
            f"{pct:.2f}%  ({n_covered:,}/{n_ice:,}, {n_missing:,} missing)",
        )

        if nan_fill:
            n_out = int((has_data & ~ice_mask).sum())
            _chk(
                "No data outside ice mask",
                n_out == 0,
                f"{n_out:,} spurious" if n_out else "",
            )
        else:
            print("  [–]  Spurious-data check skipped (fill_value ≠ NaN)")

        del has_data, covered

    # ================================================================
    # 5. 2D NaN-MASK CONSISTENCY
    # ================================================================
    _hdr("5. 2D Mask Consistency")

    ref_nan = np.isnan(dto)
    for v in sorted(exp_2d - {"dist_to_ocean"}):
        if v in present:
            _chk(
                f"{v} NaN ≡ dist_to_ocean NaN",
                np.array_equal(ref_nan, np.isnan(ds[v].compute().values)),
            )
    del dto, ref_nan

    # ================================================================
    # 6. PHYSICAL RANGE CHECKS  +  COLLECT SUMMARY STATS
    # ================================================================
    _hdr("6. Physical Range Checks (ice-shelf masked)")

    stats = {}
    print("  Computing per-variable statistics via Dask …")
    for v in sorted(present):
        masked = ds[v].where(ice_da) if ice_da is not None else ds[v]

        if v in exp_3d:
            # Batch ALL reductions into one Dask graph for a single
            # pass through the 3D array (minimises I/O).
            vmin, vmax, vmean, vcount, per_t, per_px = dask.compute(
                masked.min(),
                masked.max(),
                masked.mean(),
                masked.count(),
                masked.count(dim=("y", "x")),
                masked.count(dim="time"),
            )
            stats[v] = dict(
                vmin=float(vmin),
                vmax=float(vmax),
                vmean=float(vmean),
                vcount=int(vcount),
                per_t=per_t.values,
                per_px=per_px.values,
            )
        else:
            vmin, vmax, vmean, vcount = dask.compute(
                masked.min(),
                masked.max(),
                masked.mean(),
                masked.count(),
            )
            stats[v] = dict(
                vmin=float(vmin),
                vmax=float(vmax),
                vmean=float(vmean),
                vcount=int(vcount),
            )
        print(f"    {v:>15s}  done")

    # dist_to_ocean
    s = stats["dist_to_ocean"]
    _chk("dist_to_ocean ≥ 0", s["vmin"] >= 0, f"min = {s['vmin']:.1f} m")
    _chk(
        f"dist_to_ocean < {DIST_MAX_M / 1e3:.0f} km",
        s["vmax"] < DIST_MAX_M,
        f"max = {s['vmax'] / 1e3:.1f} km",
    )

    # clamped_depth
    s = stats["clamped_depth"]
    _chk("clamped_depth ≥ 0", s["vmin"] >= 0, f"min = {s['vmin']:.1f} m")
    if glorys_max_d is not None:
        _chk(
            f"clamped_depth ≤ GLORYS max ({glorys_max_d:.0f} m)",
            s["vmax"] <= glorys_max_d + 1.0,
            f"max = {s['vmax']:.1f} m",
        )

    # ice_draft (surface − thickness; negative for floating ice)
    s = stats["ice_draft"]
    _chk(
        "ice_draft range plausible",
        s["vmin"] > -3000 and s["vmax"] < 500,
        f"[{s['vmin']:.1f}, {s['vmax']:.1f}] m",
    )

    # thetao
    s = stats["thetao"]
    _chk(
        f"thetao ∈ [{THETA_LO}, {THETA_HI}] °C",
        s["vmin"] >= THETA_LO and s["vmax"] <= THETA_HI,
        f"[{s['vmin']:.3f}, {s['vmax']:.3f}]",
    )

    # so
    s = stats["so"]
    _chk(
        f"so ∈ [{SO_LO}, {SO_HI}] PSU",
        s["vmin"] >= SO_LO and s["vmax"] <= SO_HI,
        f"[{s['vmin']:.3f}, {s['vmax']:.3f}]",
    )

    # ================================================================
    # 7. PHYSICAL CONSISTENCY
    # ================================================================
    _hdr("7. Physical Consistency")

    # clamped_depth ≤ |ice_draft|  (clamping can only reduce depth)
    if ice_da is not None:
        n_viol = int(
            (
                (ds.clamped_depth > (np.abs(ds.ice_draft) + 0.01))
                .where(ice_da, False)
                .sum()
                .compute()
            )
        )
        _chk(
            "clamped_depth ≤ |ice_draft|",
            n_viol == 0,
            f"{n_viol:,} violations" if n_viol else "",
        )

    # mascon_id values should be integer-like (float32 encoding of ints)
    if ice_da is not None:
        msc = ds.mascon_id.where(ice_da)
        n_non_int = int(
            ((msc - msc.round()).pipe(np.abs) > 0.01).sum().compute()
        )
        _chk(
            "mascon_id integer-valued",
            n_non_int == 0,
            f"{n_non_int:,} non-integer values"
            if n_non_int
            else f"range: [{stats['mascon_id']['vmin']:.0f}, "
            f"{stats['mascon_id']['vmax']:.0f}]",
        )

    # ================================================================
    # 8. TEMPORAL INTEGRITY
    # ================================================================
    _hdr("8. Temporal Integrity")

    for v in sorted(exp_3d & present):
        s = stats[v]
        per_t = s["per_t"]      # (nT,) valid pixels per time step
        per_px = s["per_px"]    # (ny, nx) valid time steps per pixel

        # No empty time steps
        n_empty = int(np.sum(per_t == 0))
        _chk(
            f"{v}: no empty time steps",
            n_empty == 0,
            f"{n_empty} empty"
            if n_empty
            else f"valid/step: [{int(per_t.min()):,} .. {int(per_t.max()):,}]",
        )

        # Full temporal coverage at every ice-shelf pixel
        if ice_mask is not None:
            px_ice = per_px[ice_mask]
            if len(px_ice) > 0:
                pct_full = 100 * np.mean(px_ice == nT)
                mn, mx = int(px_ice.min()), int(px_ice.max())
                _chk(
                    f"{v}: full temporal coverage",
                    mn == nT,
                    f"{pct_full:.1f}% fully covered  "
                    f"[min={mn}/{nT}, max={mx}/{nT}]",
                )

    # ================================================================
    # 9. VARIABLE METADATA
    # ================================================================
    _hdr("9. Variable Metadata")

    for v in sorted(present):
        has_units = "units" in ds[v].attrs or v == "mascon_id"
        _chk(
            f"{v} has units",
            has_units,
            ds[v].attrs.get("units", "(n/a — dimensionless ID)"),
        )
        _chk(f"{v} has long_name", "long_name" in ds[v].attrs)

    # ================================================================
    # 10. SUMMARY STATISTICS TABLE
    # ================================================================
    _hdr("10. Summary Statistics (ice-shelf masked)")

    header = (
        f"  {'Variable':>15s}  {'min':>12s}  {'mean':>12s}  "
        f"{'max':>12s}  {'n_valid':>14s}"
    )
    print(f"\n{header}")
    print(f"  {'─' * 15}  {'─' * 12}  {'─' * 12}  {'─' * 12}  {'─' * 14}")
    for v in sorted(present):
        s = stats[v]
        print(
            f"  {v:>15s}  {s['vmin']:>12.3f}  {s['vmean']:>12.3f}  "
            f"{s['vmax']:>12.3f}  {s['vcount']:>14,}"
        )

    # ================================================================
    # VERDICT
    # ================================================================
    elapsed = _time.perf_counter() - wall

    print(f"\n{'=' * 72}")
    if _n_fail == 0:
        print(f" ✓  ALL {_n_pass} CHECKS PASSED  ({elapsed:.0f} s)")
    else:
        print(
            f" ✗  {_n_fail} FAILED  /  {_n_pass + _n_fail} total  "
            f"({elapsed:.0f} s)"
        )
    print(f"{'=' * 72}")

    return _n_fail == 0


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Verify step_04a output: matched_ocean_grid.zarr",
    )
    p.add_argument(
        "--output", default=OUTPUT_ZARR, help="Path to output Zarr store"
    )
    p.add_argument(
        "--bedmap", default=BEDMAP_PATH, help="Path to Bedmap3 Zarr store"
    )
    p.add_argument(
        "--ocean", default=OCEAN_PATH, help="Path to GLORYS NetCDF"
    )
    a = p.parse_args()
    sys.exit(0 if verify(a.output, a.bedmap, a.ocean) else 1)

