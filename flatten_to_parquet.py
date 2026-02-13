"""
flatten_to_parquet.py

-------------------------------------------------------------------------------
COMPUTATIONAL GLACIOLOGY - OPTIMISED PIPELINE
-------------------------------------------------------------------------------
DATE:   2026-02-13
STATUS: PRODUCTION

Flattens multidimensional Zarr stores into row-based Parquet tables optimised
for the PySpark-based Long Sparse Parquet (LSP) assembly in build_lsp.py.

Bifurcated Flattening Strategy
-------------------------------
Cartesian-joining static topography across N time steps violates database
normalisation and will OOM a 64 GB machine during PySpark shuffle.  Instead:

  1. Static 2D    Bedmap3 + mascon + spatial features + ocean 2D  -> single file
  2. ICESat-2 3D  elevation change + dynamic spatial features     -> per-step files
  3. Ocean 3D     thetao, so, T_f, T_star                        -> per-step files
  4. GRACE        mascon mass anomalies                           -> single file

Memory: peak ~12 GB per time step in the 3D loops (64 GB machine).
-------------------------------------------------------------------------------
"""

import os
import shutil
import gc
import time as _time

import numpy as np
import xarray as xr
import pyarrow as pa
import pyarrow.parquet as pq

# ── Configuration ───────────────────────────────────────────────────────────
DIR_PROCESSED = "data/processed"
DIR_LAYERS    = "data/processed_layers"
DIR_FLATTENED = "data/flattened"

# Inputs
ZARR_BEDMAP   = os.path.join(DIR_PROCESSED, "bedmap3_500m.zarr")
ZARR_MASCON   = os.path.join(DIR_PROCESSED, "mascon_id_master_map.zarr")
ZARR_ICESAT   = os.path.join(DIR_PROCESSED, "icesat2_500m_deltah.zarr")
ZARR_GRACE    = os.path.join(DIR_PROCESSED, "grace.zarr")
ZARR_FEATURES = os.path.join(DIR_PROCESSED, "spatial_features_engineered.zarr")
ZARR_OCEAN    = os.path.join(DIR_LAYERS,    "thermodynamic_ocean_grid.zarr")

# Outputs (each is a directory containing one or more Parquet files)
OUT_STATIC = os.path.join(DIR_FLATTENED, "bedmap3_static.parquet")
OUT_ICESAT = os.path.join(DIR_FLATTENED, "icesat2_dynamic.parquet")
OUT_OCEAN  = os.path.join(DIR_FLATTENED, "ocean_dynamic.parquet")
OUT_GRACE  = os.path.join(DIR_FLATTENED, "grace.parquet")

PQ_COMPRESSION = "zstd"


# ── Helpers ─────────────────────────────────────────────────────────────────

def _prep_dir(path):
    """Clear and recreate an output directory."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _validate_inputs():
    """Fail fast if any required Zarr store is missing."""
    stores = [ZARR_BEDMAP, ZARR_MASCON, ZARR_ICESAT,
              ZARR_GRACE, ZARR_FEATURES, ZARR_OCEAN]
    missing = [z for z in stores if not os.path.exists(z)]
    if missing:
        for z in missing:
            print(f"[ERROR]  Missing: {z}")
        raise FileNotFoundError(
            f"{len(missing)} required Zarr store(s) not found."
        )


# ── 1/4  Static topography + ocean 2D ──────────────────────────────────────

def flatten_static():
    """
    Flattens all strictly 2D features into a single Parquet file.

    Sources
    -------
    Bedmap3      :  surface, bed, thickness, mask
    Mascon map   :  mascon_id
    Spatial feat :  bed_slope, dist_to_grounding_line
    Ocean 2D     :  clamped_depth, dist_to_ocean, ice_draft
                    (NaN for grounded ice - only ice-shelf pixels have values)

    The full 12288x12288 grid is loaded (~10 GB combined), masked to ice
    (grounded=1, floating shelf=3), and written as a single compressed file.

    mascon_id is NOT dropped when NaN.  Pixels outside the GRACE mascon
    domain still carry valid Bedmap3, ICESat-2 and ocean data.  The GRACE
    join in build_lsp.py naturally returns null for those rows.
    """
    t0 = _time.perf_counter()
    print("\n" + "=" * 72)
    print(" 1/4  STATIC TOPOGRAPHY + OCEAN 2D")
    print("=" * 72)
    _prep_dir(OUT_STATIC)

    print("[Load]   Opening static Zarr stores ...")
    ds_bed  = xr.open_zarr(ZARR_BEDMAP)
    ds_mas  = xr.open_zarr(ZARR_MASCON)
    ds_feat = xr.open_zarr(ZARR_FEATURES)
    ds_oc   = xr.open_zarr(ZARR_OCEAN)

    # Assemble - xarray aligns on (y, x) automatically.
    # We exclude the ocean store's mascon_id to avoid conflict with
    # the canonical master mascon map.
    print("[Merge]  Assembling static Dataset ...")
    ds = xr.Dataset({
        # Bedmap3
        "surface":                ds_bed["surface"],
        "bed":                    ds_bed["bed"],
        "thickness":              ds_bed["thickness"],
        "mask":                   ds_bed["mask"],
        # Mascon
        "mascon_id":              ds_mas["mascon_id"],
        # Static spatial features
        "bed_slope":              ds_feat["bed_slope"],
        "dist_to_grounding_line": ds_feat["dist_to_grounding_line"],
        # Ocean 2D (valid for ice-shelf pixels only)
        "clamped_depth":          ds_oc["clamped_depth"],
        "dist_to_ocean":          ds_oc["dist_to_ocean"],
        "ice_draft":              ds_oc["ice_draft"],
    })

    print("[Compute] Loading full grid -> DataFrame ...")
    df = ds.compute().to_dataframe().reset_index()

    # Keep only ice pixels (grounded=1, floating shelf=3).
    # Ocean (0), exposed rock (2), etc. are dropped.
    n_total = len(df)
    df = df[df["mask"].isin([1, 3])]
    n_ice = len(df)
    print(f"  -> {n_total:,} total -> {n_ice:,} ice rows "
          f"({n_ice / n_total * 100:.1f}%)")

    print(f"[Write]  {OUT_STATIC}")
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        os.path.join(OUT_STATIC, "part-00000.parquet"),
        compression=PQ_COMPRESSION,
    )

    del ds, df
    gc.collect()
    print(f"[Done]   [{_time.perf_counter() - t0:.1f} s]")


# ── 2/4  ICESat-2 dynamic kinematics ───────────────────────────────────────

def flatten_icesat_dynamic():
    """
    Flattens 3D kinematics one time step at a time.

    Per step: isel -> mask to ice -> to_dataframe -> dropna(delta_h) -> write.
    Rows where ICESat-2 had no observation (NaN delta_h) are dropped -
    this is the 'sparse' in Long Sparse Parquet.
    """
    t0 = _time.perf_counter()
    print("\n" + "=" * 72)
    print(" 2/4  ICESAT-2 DYNAMIC KINEMATICS")
    print("=" * 72)
    _prep_dir(OUT_ICESAT)

    ds_is   = xr.open_zarr(ZARR_ICESAT)
    ds_feat = xr.open_zarr(ZARR_FEATURES)

    # Pre-load the ice mask once (grounded=1, floating shelf=3)
    ice_mask = xr.open_zarr(ZARR_BEDMAP)["mask"].isin([1, 3]).compute()

    print("[Merge]  Aligning dynamic datasets ...")
    ds = xr.Dataset({
        "delta_h":           ds_is["delta_h"],
        "ice_area":          ds_is["ice_area"],
        "h_surface_dynamic": ds_feat["h_surface_dynamic"],
        "surface_slope":     ds_feat["surface_slope"],
    })

    times = ds.time.values
    nT = len(times)
    print(f"[Iterate] {nT} time steps\n")

    for i, t in enumerate(times):
        ts = _time.perf_counter()

        # isel drops the time dim; time becomes a scalar coordinate.
        # where() sets non-ice pixels to NaN; the full grid is still
        # 12288x12288 - actual row reduction happens at the dropna step.
        ds_t = ds.isel(time=i).where(ice_mask)
        df   = ds_t.compute().to_dataframe().reset_index()

        # Guarantee the time column exists (scalar coords are normally
        # broadcast by to_dataframe, but be explicit for robustness).
        if "time" not in df.columns:
            df["time"] = t

        # Drop rows with no ICESat-2 observation (the 'sparse' in LSP).
        df = df.dropna(subset=["delta_h"])
        n = len(df)

        pq.write_table(
            pa.Table.from_pandas(df, preserve_index=False),
            os.path.join(OUT_ICESAT, f"step_{i:03d}.parquet"),
            compression=PQ_COMPRESSION,
        )

        del ds_t, df
        gc.collect()
        print(f"  step {i + 1:>3}/{nT}  |  {str(t)[:10]}  "
              f"|  {n:>10,} rows  |  {_time.perf_counter() - ts:.1f} s")

    print(f"\n[Done]   [{_time.perf_counter() - t0:.1f} s]")


# ── 3/4  Ocean thermodynamics ──────────────────────────────────────────────

def flatten_ocean_dynamic():
    """
    Flattens the 3D ocean variables (thetao, so, T_f, T_star) from the
    thermodynamic ocean grid, one time step at a time.

    Only ice-shelf pixels with valid ocean interpolation are kept (dropna
    on thetao).  Grounded-ice and open-ocean pixels are NaN in the source
    store and are naturally excluded.
    """
    t0 = _time.perf_counter()
    print("\n" + "=" * 72)
    print(" 3/4  OCEAN THERMODYNAMICS")
    print("=" * 72)
    _prep_dir(OUT_OCEAN)

    ds_oc = xr.open_zarr(ZARR_OCEAN)
    times = ds_oc.time.values
    nT = len(times)
    print(f"[Iterate] {nT} time steps\n")

    for i, t in enumerate(times):
        ts = _time.perf_counter()

        # Select only the 3D ocean variables for this time step.
        # isel drops the time dim; we re-attach it as a scalar coord
        # so it appears in the DataFrame for the downstream time join.
        ds_t = xr.Dataset({
            "thetao": ds_oc["thetao"].isel(time=i),
            "so":     ds_oc["so"].isel(time=i),
            "T_f":    ds_oc["T_f"].isel(time=i),
            "T_star": ds_oc["T_star"].isel(time=i),
        })

        df = ds_t.compute().to_dataframe().reset_index()

        # Ensure time column is present (scalar coord should broadcast,
        # but be explicit for robustness).
        if "time" not in df.columns:
            df["time"] = t

        # Drop pixels with no ocean data at this time step.
        df = df.dropna(subset=["thetao"])
        n = len(df)

        if n > 0:
            pq.write_table(
                pa.Table.from_pandas(df, preserve_index=False),
                os.path.join(OUT_OCEAN, f"step_{i:03d}.parquet"),
                compression=PQ_COMPRESSION,
            )

        del ds_t, df
        gc.collect()
        print(f"  step {i + 1:>3}/{nT}  |  {str(t)[:10]}  "
              f"|  {n:>10,} rows  |  {_time.perf_counter() - ts:.1f} s")

    print(f"\n[Done]   [{_time.perf_counter() - t0:.1f} s]")


# ── 4/4  GRACE mass anomalies ──────────────────────────────────────────────

def flatten_grace():
    """
    Flattens the GRACE mascon mass-anomaly time series.

    The dataset is small enough (~hundreds of mascons x hundreds of months)
    to load entirely in memory.  Only rows with valid lwe_length are kept.
    """
    t0 = _time.perf_counter()
    print("\n" + "=" * 72)
    print(" 4/4  GRACE MASS ANOMALIES")
    print("=" * 72)
    _prep_dir(OUT_GRACE)

    ds = xr.open_zarr(ZARR_GRACE)
    df = ds.compute().to_dataframe().reset_index()
    df = df.dropna(subset=["lwe_length"])

    print(f"  -> {len(df):,} valid mascon-month rows")

    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        os.path.join(OUT_GRACE, "part-00000.parquet"),
        compression=PQ_COMPRESSION,
    )

    print(f"[Done]   [{_time.perf_counter() - t0:.1f} s]")


# ── Entrypoint ──────────────────────────────────────────────────────────────

def main():
    wall = _time.perf_counter()
    print("=" * 72)
    print(" PARQUET FLATTENING PIPELINE")
    print("=" * 72)

    _validate_inputs()
    os.makedirs(DIR_FLATTENED, exist_ok=True)

    flatten_static()
    flatten_icesat_dynamic()
    flatten_ocean_dynamic()
    flatten_grace()

    print(f"\n{'=' * 72}")
    print(f" ALL DONE  [{_time.perf_counter() - wall:.0f} s  "
          f"({((_time.perf_counter() - wall) / 60):.1f} min)]")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
