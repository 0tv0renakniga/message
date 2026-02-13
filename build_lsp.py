"""
build_lsp.py

-------------------------------------------------------------------------------
COMPUTATIONAL GLACIOLOGY - OPTIMISED PIPELINE
-------------------------------------------------------------------------------
DATE:   2026-02-13
STATUS: PRODUCTION

Assembles the Antarctica Digital Twin Long Sparse Parquet (LSP) from the
four flat Parquet tables produced by flatten_to_parquet.py.

Architecture
------------

  ICESat-2 dynamic ----+                     +---- ocean dynamic
                       |  FULL OUTER JOIN    |
                       +--  (y, x, time)  ---+
                                |
                         temporal backbone
                                |
                         LEFT JOIN (y, x)  <---- static features
                                |
                    LEFT JOIN (mascon_id, time)  <---- GRACE anomalies
                                |
                        forward fill per pixel
                                |
                     Constrained Forward Modeling
                   (GRACE x ICESat-2 signal fusion)
                                |
                           write LSP

Join Logic
----------
1. FULL OUTER JOIN ICESat-2 and ocean dynamic on (y, x, time).
   Creates the temporal backbone: a row exists for every (pixel, time)
   where at least one sensor has an observation.

2. LEFT JOIN static features on (y, x).
   Enriches every temporal row with Bedmap3 topography, mascon ID,
   spatial features, and ocean 2D properties.

3. LEFT JOIN GRACE anomalies on (mascon_id, time).
   Adds the coarse mass-change signal.  mascon_id is cast to integer
   so NaN (float) becomes null -> no GRACE match -> lwe_length is null.

Forward Fill
------------
Window.partitionBy("y", "x").orderBy("time") carries the last valid
observation forward for each temporal column independently.  This fills
gaps where one sensor has data but another does not at that time step.
Pixels with no preceding observation remain null (no backward fill).

Signal Fusion (Constrained Forward Modeling)
--------------------------------------------
Downscales the coarse GRACE mascon-level mass change to the 500 m pixel
level using the high-resolution ICESat-2 elevation-change pattern as a
spatial template.

    Forward model:   dm_forward_i = dh_i * rho_ice           (per pixel)
    Constraint:      SUM(dm_fused_i * A_i) = DM_GRACE        (per mascon)
    Scale factor:    alpha = DM_GRACE / SUM(dm_forward_i * A_i)
    Fused signal:    lwe_fused_i = (dh_i / SUM(dh_j)) * N * lwe_length

The ice density cancels in the normalisation, so the result depends only
on the relative dh pattern and the GRACE total.  Verification:

    area-weighted mean of lwe_fused = lwe_length   (exact for uniform grid)

A uniform fallback is applied when the mascon-mean |dh| is below the
noise floor (0.1 mm), since the ICESat-2 pattern cannot be trusted.

Type Safety
-----------
NumPy datetime64[ns] -> PyArrow TIMESTAMP(NANOS) in Parquet.  PySpark's
TimestampType is microsecond precision.  The _spark_safe_types() helper
casts any non-TimestampType 'time' column before joins, and the Spark
session writes TIMESTAMP_MICROS to the output LSP.

Memory
------
Configured for a single-node 64 GB machine using Spark local[*] mode
with 48 GB driver memory.  Adaptive Query Execution (AQE) dynamically
tunes shuffle partitions at runtime.
-------------------------------------------------------------------------------
"""

import os
import time as _time

from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, TimestampType

# ── Configuration ───────────────────────────────────────────────────────────
DIR_FLATTENED = "data/flattened"

STATIC_PATH = os.path.join(DIR_FLATTENED, "bedmap3_static.parquet")
ICESAT_PATH = os.path.join(DIR_FLATTENED, "icesat2_dynamic.parquet")
OCEAN_PATH  = os.path.join(DIR_FLATTENED, "ocean_dynamic.parquet")
GRACE_PATH  = os.path.join(DIR_FLATTENED, "grace.parquet")

LSP_PATH = os.path.join(DIR_FLATTENED, "antarctica_lsp.parquet")

# Temporal columns to forward-fill per pixel.
# Only columns that actually exist in the joined DataFrame are processed.
# NOTE: lwe_fused is intentionally ABSENT -- it is a derived quantity
#       computed from already-forward-filled delta_h and lwe_length.
FFILL_COLS = [
    "delta_h", "ice_area", "h_surface_dynamic", "surface_slope",
    "thetao", "so", "T_f", "T_star", "lwe_length",
]

# ── Signal Fusion constants ─────────────────────────────────────────────────
#
# Minimum mascon-mean |dh| (m) below which ICESat-2 cannot resolve the
# spatial pattern and GRACE is distributed uniformly.
# 0.1 mm is well below altimetric noise (~2 cm) -- this only catches
# true numerical-zero cases.
FUSION_MIN_MEAN_DH_M = 1e-4  # 0.1 mm

DRIVER_MEMORY = "48g"


# ── Helpers ─────────────────────────────────────────────────────────────────

def _validate_inputs():
    """Fail fast if any required Parquet directory is missing."""
    paths = [STATIC_PATH, ICESAT_PATH, OCEAN_PATH, GRACE_PATH]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"[ERROR]  Missing: {p}")
        raise FileNotFoundError(
            f"{len(missing)} required Parquet source(s) not found.  "
            "Run flatten_to_parquet.py first."
        )


def _spark_safe_types(df, name=""):
    """
    Coerce DataFrame columns to PySpark-native types before joins.

    Primary concern
    ---------------
    NumPy datetime64[ns]  ->  PyArrow TIMESTAMP(NANOS) in Parquet.
    PySpark's TimestampType is microsecond precision.  On Spark < 3.4,
    nanosecond Parquet timestamps can cause ArrowException or silent
    truncation.  This function casts any non-TimestampType ``time``
    column to TimestampType (us), providing a safe read path regardless
    of how the upstream Parquet was written.

    All other NumPy dtypes (float32/64, int32/64, bool) map cleanly
    from PyArrow -> Spark with no conversion needed.
    """
    for field in df.schema.fields:
        if field.name == "time" and not isinstance(field.dataType, TimestampType):
            old_type = field.dataType
            df = df.withColumn("time", F.to_timestamp(F.col("time")))
            if name:
                print(f"  [{name}] time: {old_type} -> TimestampType (us)")
    return df


def _fuse_grace_icesat(df):
    """
    Constrained Forward Modeling: downscale GRACE mascon-level mass change
    to the 500 m pixel level using ICESat-2 dh as the spatial template.

    Physics
    -------
    GRACE-FO measures total mass change (DM) at ~200 km mascon resolution.
    ICESat-2 measures surface elevation change (dh) at ~500 m resolution.
    Converting dh to mass (dm = dh * rho_ice) gives a "forward model" of
    what GRACE should observe.  The ratio alpha = DM_GRACE / DM_forward is
    the constrained scale factor.

    After normalisation, rho_ice cancels:

        w_i = dh_i / SUM_mascon(dh_j)        (signed weight, sums to 1.0)
        lwe_fused_i = w_i * N * lwe_length    (per-pixel LWE, same units)

    Verification:  area-weighted mean of lwe_fused == lwe_length  (exact
    for the uniform 500 m EPSG:3031 grid).

    Fallback
    --------
    When mascon-mean |dh| < FUSION_MIN_MEAN_DH_M (0.1 mm), ICESat-2
    cannot resolve the spatial pattern.  The GRACE signal is distributed
    uniformly: lwe_fused_i = lwe_length.

    Guard rails
    -----------
    - Pixels with null mascon_id  -> lwe_fused = null  (no GRACE data)
    - Pixels with null delta_h    -> lwe_fused = null  (no pattern)
    - Pixels with null lwe_length -> lwe_fused = null  (no GRACE data)

    Parameters (implicit via DataFrame columns)
    --------------------------------------------
    delta_h     : ICESat-2 elevation change (m), forward-filled
    lwe_length  : GRACE mascon LWE anomaly (mm w.e.), forward-filled
    mascon_id   : mascon identifier (integer)
    time        : temporal coordinate

    Returns
    -------
    DataFrame with added ``lwe_fused`` column (same units as lwe_length).
    Intermediate columns are dropped.
    """
    w_mt = Window.partitionBy("mascon_id", "time")

    # ── Mascon-level aggregates ────────────────────────────────────────
    # N  = count of non-null delta_h in this mascon at this time step.
    # Σh = sum of signed delta_h (normalisation denominator).
    df = df.withColumn("_n_pix",  F.count("delta_h").over(w_mt))
    df = df.withColumn("_sum_dh", F.sum("delta_h").over(w_mt))

    # ── Pattern vs uniform decision ───────────────────────────────────
    # If the mean signed |dh| in the mascon exceeds the noise floor,
    # the ICESat-2 spatial pattern is trustworthy.
    # |Σ dh / N| > threshold   =>   pattern mode
    # otherwise                 =>   uniform fallback
    use_pattern = (
        F.abs(F.col("_sum_dh") / F.col("_n_pix"))
        > F.lit(FUSION_MIN_MEAN_DH_M)
    )

    # ── Fused LWE per pixel ───────────────────────────────────────────
    # Pattern:  lwe_fused = (dh_i / Σ dh_j) × N × lwe_length
    # Uniform:  lwe_fused = lwe_length  (every pixel gets mascon average)
    # Null:     lwe_fused = null        (missing GRACE or ICESat-2)
    pattern_lwe = (
        (F.col("delta_h") / F.col("_sum_dh"))
        * F.col("_n_pix")
        * F.col("lwe_length")
    )

    has_data = (
        F.col("mascon_id").isNotNull()
        & F.col("delta_h").isNotNull()
        & F.col("lwe_length").isNotNull()
    )

    df = df.withColumn(
        "lwe_fused",
        F.when(
            has_data,
            F.when(use_pattern, pattern_lwe).otherwise(F.col("lwe_length"))
        )
    )

    # ── Cleanup intermediate columns ──────────────────────────────────
    df = df.drop("_n_pix", "_sum_dh")

    return df


# ── Pipeline ────────────────────────────────────────────────────────────────

def main():
    wall = _time.perf_counter()

    _validate_inputs()

    # ── Spark session ────────────────────────────────────────────────────
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("Antarctica Digital Twin - LSP Assembly")
        .config("spark.driver.memory", DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.parquet.compression.codec", "zstd")
        # Write timestamps as TIMESTAMP_MICROS (not legacy INT96).
        # Ensures the output LSP is natively readable by any modern
        # Parquet consumer without INT96-to-us rebasing.
        .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print("=" * 72)
    print(" BUILD LONG SPARSE PARQUET (LSP)")
    print("=" * 72)

    # ── 1. Read flat Parquet sources ─────────────────────────────────────
    print("\n[Read]   Loading Parquet sources ...")

    df_static = spark.read.parquet(STATIC_PATH)
    df_icesat = spark.read.parquet(ICESAT_PATH)
    df_ocean  = spark.read.parquet(OCEAN_PATH)
    df_grace  = spark.read.parquet(GRACE_PATH)

    for name, df in [("static",  df_static), ("icesat2", df_icesat),
                     ("ocean",   df_ocean),  ("grace",   df_grace)]:
        ncol = len(df.columns)
        print(f"  -> {name:8s}  {ncol} cols  {df.columns}")

    # ── 2. Sanitise types (ns -> us timestamps) ──────────────────────────
    #
    # NumPy datetime64[ns] may arrive as TIMESTAMP(NANOS) in Parquet.
    # PySpark's TimestampType is us-precision.  Explicit casting prevents
    # ArrowException on Spark < 3.4 and silent truncation everywhere.
    #
    print("\n[Types]  Sanitising data types ...")
    df_icesat = _spark_safe_types(df_icesat, "icesat2")
    df_ocean  = _spark_safe_types(df_ocean,  "ocean")
    df_grace  = _spark_safe_types(df_grace,  "grace")

    # ── 3. Temporal backbone: ICESat-2 FULL OUTER JOIN ocean ─────────────
    #
    # A row exists for every (y, x, time) where at least one sensor
    # produced a valid observation.  Columns from the other sensor are
    # null where it had no data.
    #
    print("\n[Join 1] ICESat-2  FULL OUTER JOIN  ocean  on (y, x, time)")
    t0 = _time.perf_counter()
    df_temporal = df_icesat.join(
        df_ocean, on=["y", "x", "time"], how="full_outer"
    )
    print(f"         schema: {df_temporal.columns}")
    print(f"         [{_time.perf_counter() - t0:.1f} s  (lazy)]")

    # ── 4. Enrich with static features ───────────────────────────────────
    #
    # LEFT JOIN on (y, x) adds Bedmap3 topography, mascon_id, spatial
    # features, and ocean 2D properties to every temporal row.
    #
    print("[Join 2] LEFT JOIN  static  on (y, x)")
    t0 = _time.perf_counter()
    df = df_temporal.join(df_static, on=["y", "x"], how="left")
    print(f"         [{_time.perf_counter() - t0:.1f} s  (lazy)]")

    # ── 5. Enrich with GRACE mass anomalies ──────────────────────────────
    #
    # Cast mascon_id to integer in both tables so that:
    #   - NaN (float) becomes null -> no GRACE match
    #   - Avoids float-precision mismatches in the equi-join
    #
    # Select only the join key + payload from GRACE to prevent
    # column collisions (GRACE may carry lat/lon/area columns).
    #
    print("[Join 3] LEFT JOIN  GRACE  on (mascon_id, time)")
    t0 = _time.perf_counter()
    df = df.withColumn("mascon_id", F.col("mascon_id").cast(IntegerType()))

    grace_cols = [c for c in ["mascon_id", "time", "lwe_length"]
                  if c in df_grace.columns]
    df_grace_clean = (
        df_grace
        .select(grace_cols)
        .withColumn("mascon_id", F.col("mascon_id").cast(IntegerType()))
    )
    df_grace_clean = _spark_safe_types(df_grace_clean, "grace_join")

    df = df.join(df_grace_clean, on=["mascon_id", "time"], how="left")
    print(f"         [{_time.perf_counter() - t0:.1f} s  (lazy)]")

    # ── 6. Forward fill temporal columns per pixel ───────────────────────
    #
    # For each ice pixel, order by time and carry the last non-null
    # observation forward.  This fills gaps where sensor A has data
    # but sensor B does not at that time step.
    #
    # Pixels with no preceding observation remain null (no backward fill).
    #
    print("\n[FFill]  Forward-filling temporal columns ...")
    t0 = _time.perf_counter()

    w = (
        Window.partitionBy("y", "x")
        .orderBy("time")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )

    existing = set(df.columns)
    for col_name in FFILL_COLS:
        if col_name in existing:
            df = df.withColumn(
                col_name, F.last(col_name, ignorenulls=True).over(w)
            )
            print(f"  -> {col_name}")

    print(f"         [{_time.perf_counter() - t0:.1f} s  (lazy)]")

    # ── 7. Signal Fusion: Constrained Forward Modeling ───────────────────
    #
    # Downscale the coarse GRACE mascon-level mass change to the 500 m
    # pixel level using the forward-filled ICESat-2 dh pattern.
    #
    # lwe_fused_i = (dh_i / SUM(dh_j)) * N * lwe_length
    #
    # This runs AFTER forward fill so that fusion operates on the best
    # available estimate at each time step.  lwe_fused itself is NOT
    # forward-filled because it is already computed at every time step.
    #
    print("\n[Fuse]   Constrained Forward Modeling (GRACE x ICESat-2) ...")
    t0 = _time.perf_counter()
    df = _fuse_grace_icesat(df)
    print(f"  -> lwe_fused  (noise floor = {FUSION_MIN_MEAN_DH_M} m)")
    print(f"         [{_time.perf_counter() - t0:.1f} s  (lazy)]")

    # ── 8. Materialise and write LSP ─────────────────────────────────────
    #
    # repartition(200) creates ~200 evenly-sized output files for
    # balanced downstream reads.  zstd compression is set at the
    # session level via spark.sql.parquet.compression.codec.
    #
    print(f"\n[Write]  {LSP_PATH}")
    t0 = _time.perf_counter()

    (
        df
        .repartition(200)
        .write
        .mode("overwrite")
        .parquet(LSP_PATH)
    )

    print(f"[Write]  Done  [{_time.perf_counter() - t0:.1f} s]")

    # ── Summary ──────────────────────────────────────────────────────────
    wall_t = _time.perf_counter() - wall
    print(f"\n{'=' * 72}")
    print(f" LSP ASSEMBLY COMPLETE  [{wall_t:.0f} s  ({wall_t / 60:.1f} min)]")
    print(f"{'=' * 72}")

    spark.stop()


if __name__ == "__main__":
    main()
