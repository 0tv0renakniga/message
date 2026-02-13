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
FFILL_COLS = [
    "delta_h", "ice_area", "h_surface_dynamic", "surface_slope",
    "thetao", "so", "T_f", "T_star", "lwe_length",
]

DRIVER_MEMORY = "48g"


# ── Helpers ─────────────────────────────────────────────────────────────────

def _ensure_timestamp(df, col="time"):
    """Cast a column to TimestampType if it is not already."""
    field = [f for f in df.schema.fields if f.name == col]
    if field and not isinstance(field[0].dataType, TimestampType):
        df = df.withColumn(col, F.to_timestamp(col))
    return df


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

    # ── 2. Normalise time columns ────────────────────────────────────────
    #
    # Depending on how PyArrow serialised the timestamps, Spark may have
    # read them as strings.  Cast to TimestampType for correct joins.
    #
    df_icesat = _ensure_timestamp(df_icesat, "time")
    df_ocean  = _ensure_timestamp(df_ocean,  "time")
    df_grace  = _ensure_timestamp(df_grace,  "time")

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
    df_grace_clean = _ensure_timestamp(df_grace_clean, "time")

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

    # ── 7. Materialise and write LSP ─────────────────────────────────────
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
