import xarray as xr
import numpy as np
import sys
from glob import glob


def inspect_icesat_tiles(filepaths):
    """Compare coordinate systems across ICESat-2 tiles"""
    print("=" * 80)
    print("🔍 COMPARING ICESAT-2 TILES")
    print("=" * 80)
    
    tile_info = []
    
    for filepath in sorted(filepaths):
        print(f"\n📁 {filepath.split('/')[-1]}")
        print("-" * 80)
        
        ds = xr.open_dataset(filepath, group='delta_h')
        
        info = {
            'file': filepath.split('/')[-1],
            'tile_id': filepath.split('_')[1],  # A1, A2, A3, A4
            'x_min': float(ds.x.min()),
            'x_max': float(ds.x.max()),
            'y_min': float(ds.y.min()),
            'y_max': float(ds.y.max()),
            'x_len': len(ds.x),
            'y_len': len(ds.y),
            'x_spacing': float(np.diff(ds.x.values[:10]).mean()),
            'y_spacing': float(np.diff(ds.y.values[:10]).mean()),
        }
        
        tile_info.append(info)
        
        print(f"  Tile: {info['tile_id']}")
        print(f"  X: {info['x_min']:>12,.0f} to {info['x_max']:>12,.0f}  ({info['x_len']:>5,} cells)")
        print(f"  Y: {info['y_min']:>12,.0f} to {info['y_max']:>12,.0f}  ({info['y_len']:>5,} cells)")
        print(f"  Resolution: {info['x_spacing']:.0f}m × {abs(info['y_spacing']):.0f}m")
        print(f"  X first 3: {ds.x.values[:3]}")
        print(f"  Y first 3: {ds.y.values[:3]}")
        
        ds.close()
    
    # Detailed overlap analysis
    print("\n" + "=" * 80)
    print("📊 OVERLAP ANALYSIS")
    print("=" * 80)
    
    # Sort by Y coordinate (bottom to top)
    tile_info.sort(key=lambda x: x['y_min'])
    
    for i, tile1 in enumerate(tile_info):
        print(f"\n🗺️  {tile1['tile_id']}: Y=[{tile1['y_min']:,.0f}, {tile1['y_max']:,.0f}], X=[{tile1['x_min']:,.0f}, {tile1['x_max']:,.0f}]")
        
        for j, tile2 in enumerate(tile_info):
            if i >= j:
                continue
            
            # Calculate overlap in X and Y
            x_overlap_start = max(tile1['x_min'], tile2['x_min'])
            x_overlap_end = min(tile1['x_max'], tile2['x_max'])
            x_overlap = x_overlap_end - x_overlap_start
            
            y_overlap_start = max(tile1['y_min'], tile2['y_min'])
            y_overlap_end = min(tile1['y_max'], tile2['y_max'])
            y_overlap = y_overlap_end - y_overlap_start
            
            if x_overlap > 0 and y_overlap > 0:
                print(f"  ✓ OVERLAPS {tile2['tile_id']}:")
                print(f"      X overlap: {x_overlap:>8,.0f}m  ({x_overlap_start:,.0f} to {x_overlap_end:,.0f})")
                print(f"      Y overlap: {y_overlap:>8,.0f}m  ({y_overlap_start:,.0f} to {y_overlap_end:,.0f})")
                print(f"      Overlap cells: {int(x_overlap/1000)} × {int(y_overlap/1000)} ≈ {int(x_overlap/1000 * y_overlap/1000):,} pixels")
            else:
                # Calculate gap
                if x_overlap <= 0:
                    x_gap = abs(x_overlap)
                    print(f"  ⚠️  GAP from {tile2['tile_id']} in X: {x_gap:,.0f}m")
                if y_overlap <= 0:
                    y_gap = abs(y_overlap)
                    print(f"  ⚠️  GAP from {tile2['tile_id']} in Y: {y_gap:,.0f}m")
    
    print("\n" + "=" * 80)


def inspect_netcdf(filepath):
    """Comprehensive NetCDF file inspector"""
    
    print("=" * 80)
    print(f"📁 FILE: {filepath}")
    print("=" * 80)
    
    try:
        # Open dataset
        ds = xr.open_dataset(filepath)
        
        # ===== DIMENSIONS =====
        print("\n📏 DIMENSIONS:")
        print("-" * 80)
        for dim_name, dim_size in ds.dims.items():
            print(f"  {dim_name:20s} : {dim_size:,} elements")
        
        total_dims = np.prod(list(ds.dims.values()))
        print(f"\n  Total grid points: {total_dims:,}")
        
        # ===== COORDINATES =====
        print("\n🗺️  COORDINATES:")
        print("-" * 80)
        for coord_name in ds.coords:
            coord = ds.coords[coord_name]
            print(f"  {coord_name:20s} : {coord.dims} → {coord.dtype} | Range: [{coord.min().values:.4f}, {coord.max().values:.4f}]")
        
        # ===== VARIABLES =====
        print("\n📊 DATA VARIABLES:")
        print("-" * 80)
        for var_name in ds.data_vars:
            var = ds[var_name]
            shape_str = " × ".join([f"{ds.dims[d]:,}" for d in var.dims])
            size_bytes = var.nbytes
            size_mb = size_bytes / (1024**2)
            
            print(f"\n  Variable: {var_name}")
            print(f"    Dimensions: {var.dims}")
            print(f"    Shape:      {shape_str}")
            print(f"    Data Type:  {var.dtype}")
            print(f"    Size:       {size_mb:.2f} MB ({size_bytes:,} bytes)")
            
            # Show key attributes
            if var.attrs:
                print(f"    Attributes:")
                for attr_key in ['units', 'long_name', 'standard_name', '_FillValue', 'missing_value']:
                    if attr_key in var.attrs:
                        print(f"      {attr_key}: {var.attrs[attr_key]}")
        
        # ===== GLOBAL ATTRIBUTES =====
        print("\n🌐 GLOBAL ATTRIBUTES:")
        print("-" * 80)
        if ds.attrs:
            for key, value in ds.attrs.items():
                # Truncate long values
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:97] + "..."
                print(f"  {key:25s} : {value_str}")
        else:
            print("  (none)")
        
        # ===== MEMORY ESTIMATE =====
        print("\n💾 MEMORY ESTIMATE:")
        print("-" * 80)
        total_size = sum([ds[var].nbytes for var in ds.data_vars])
        print(f"  Total data size:     {total_size / (1024**2):.2f} MB")
        print(f"  Total data size:     {total_size / (1024**3):.2f} GB")
        print(f"  Recommended chunk:   {total_size / (1024**2) / 100:.2f} MB per chunk")
        
        # ===== CRS INFO (if present) =====
        print("\n🗺️  SPATIAL REFERENCE:")
        print("-" * 80)
        crs_found = False
        for var_name in list(ds.data_vars) + list(ds.coords):
            var = ds[var_name]
            if 'crs' in var.attrs or 'grid_mapping' in var.attrs:
                print(f"  Variable '{var_name}' has CRS info:")
                print(f"    {var.attrs.get('crs', var.attrs.get('grid_mapping', 'N/A'))}")
                crs_found = True
        
        if not crs_found:
            print("  (No explicit CRS found in attributes)")
        
        ds.close()
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return
    
    print("\n" + "=" * 80)


def inspect_netcdf_with_groups(filepath):
    """Inspector that also checks for HDF5 groups"""
    
    print("=" * 80)
    print(f"📁 FILE (with group support): {filepath}")
    print("=" * 80)
    
    try:
        import h5py
        
        with h5py.File(filepath, 'r') as f:
            print("\n📦 HDF5 GROUPS:")
            print("-" * 80)
            
            def print_structure(name, obj):
                indent = "  " * name.count('/')
                if isinstance(obj, h5py.Group):
                    print(f"{indent}📁 {name}/")
                elif isinstance(obj, h5py.Dataset):
                    shape_str = " × ".join([str(s) for s in obj.shape])
                    size_mb = obj.nbytes / (1024**2)
                    print(f"{indent}📄 {name} : {shape_str} | {obj.dtype} | {size_mb:.2f} MB")
            
            f.visititems(print_structure)
        
        print("\n(Now opening with xarray for detailed view...)\n")
        inspect_netcdf(filepath)
        
    except ImportError:
        print("\n⚠️  h5py not available, using xarray only\n")
        inspect_netcdf(filepath)
    except Exception as e:
        print(f"\n⚠️  Could not read as HDF5: {e}")
        print("Trying with xarray...\n")
        inspect_netcdf(filepath)


if __name__ == "__main__":
    # Usage: python inspect_netcdf.py <filepath>
    # Or just run and it will prompt you
    """
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = input("Enter NetCDF file path: ").strip()
    
    # Check if file has groups (use advanced inspector)
    inspect_netcdf_with_groups(filepath)
    """
    files = glob("data/raw/icesat/*.nc")
    inspect_icesat_tiles(files)

    #for file_i in files:
        #inspect_netcdf_with_groups(file_i)
    #inspect_netcdf("data/processed_layers/master_grid_template.nc")
    
