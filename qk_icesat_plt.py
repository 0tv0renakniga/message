import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

files = [
    "data/raw/icesat/ATL15_A1_0328_01km_005_01.nc",
    "data/raw/icesat/ATL15_A2_0328_01km_005_01.nc",
    "data/raw/icesat/ATL15_A3_0328_01km_005_01.nc",
    "data/raw/icesat/ATL15_A4_0328_01km_005_01.nc"
]

fig, ax = plt.subplots(figsize=(14, 12))

for file in files:
    ds = xr.open_dataset(file, group='delta_h')
    
    # Get first time slice
    data = ds['ice_area'].isel(time=0)
    mask = (~data.isnull()).astype(int)
    
    # Create pixel EDGES (shift by half pixel = 500m)
    x_centers = ds.x.values
    y_centers = ds.y.values
    
    dx = 1000  # 1km resolution
    dy = 1000
    
    # Edges extend ±500m from centers
    x_edges = np.concatenate([x_centers - dx/2, [x_centers[-1] + dx/2]])
    y_edges = np.concatenate([y_centers - dy/2, [y_centers[-1] + dy/2]])
    
    # Plot with pcolormesh (respects pixel edges)
    mesh = ax.pcolormesh(x_edges, y_edges, mask.values, 
                         cmap='Greys', alpha=0.7, 
                         shading='flat',  # Use explicit edges
                         vmin=0, vmax=1)
    
    # Add tile boundary
    x_min, x_max = x_edges[0], x_edges[-1]
    y_min, y_max = y_edges[0], y_edges[-1]
    
    ax.plot([x_min, x_max, x_max, x_min, x_min],
            [y_min, y_min, y_max, y_max, y_min],
            'r-', linewidth=2, alpha=0.8)
    
    # Label
    tile_id = file.split('_')[1]
    ax.text((x_min + x_max)/2, (y_min + y_max)/2, 
            tile_id, fontsize=24, ha='center', va='center',
            color='red', weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ds.close()

ax.set_xlabel('X (meters, EPSG:3031)', fontsize=14, weight='bold')
ax.set_ylabel('Y (meters, EPSG:3031)', fontsize=14, weight='bold')
ax.set_title('ICESat-2 Tiles - Edge-Aligned', fontsize=16, weight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='cyan', linewidth=2.5, linestyle='--', alpha=0.7, label='Y=0')
ax.axvline(0, color='cyan', linewidth=2.5, linestyle='--', alpha=0.7, label='X=0')
ax.legend(fontsize=12)
ax.set_aspect('equal')  # ← Important for polar projection!

plt.tight_layout()
plt.savefig('icesat_tiles_seamless.png', dpi=150, bbox_inches='tight')
#plt.show()

