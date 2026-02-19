import xarray as xr
import numpy as np
import os

DIR = "/mnt/data/sonia/cyclone/0.25/"
# Map folder names to the variable(s) inside the files
# Note: "wind" now points to a list of variables
VARS_TO_PROCESS = {
    "geopotential": ["z"],
    "temperature_2m": ["t2m"],
    "wind_500hpa": ["u", "v"],
    "temperature": ["t"],
    "humidity": ["q"] 
}
TRAIN_YEARS = range(1979, 2016) 

means, stds = {}, {}

for folder, var_names in VARS_TO_PROCESS.items():
    file_paths = [os.path.join(DIR, folder, f"{folder}.{year}.nc") for year in TRAIN_YEARS]
    ds = xr.open_mfdataset(file_paths, combine='by_coords')
    
    for v in var_names:
        print(f"Calculating stats for {v} in {folder}...")
        # Handle 3D variables (z, q, u, v at levels) vs 2D (t2m)
        if 'pressure_level' in ds[v].dims:
            for lvl in ds.pressure_level.values:
                key = f"{v}_{int(lvl)}"
                data_lvl = ds[v].sel(pressure_level=lvl)
                means[key] = data_lvl.mean().values.item()
                stds[key] = data_lvl.std().values.item()
        else:
            means[v] = ds[v].mean().values.item()
            stds[v] = ds[v].std().values.item()

np.savez("normalize_mean.npz", **means)
np.savez("normalize_std.npz", **stds)
print("Saved normalize_mean.npz and normalize_std.npz")
