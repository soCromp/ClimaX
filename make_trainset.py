# svd env
import xarray as xr
import os 
import numpy as np
from tqdm import tqdm
import pandas as pd 
import cdsapi


variables = {'2m_temperature': ('/mnt/data/sonia/cyclone/0.25/temperature_2m/temperature_2m', 't2m', None),
             'geopotential': ('/mnt/data/sonia/cyclone/0.25/geopotential/geopotential', 'z', 925), # 925
             'u_component_of_wind': ('/mnt/data/sonia/cyclone/0.25/wind_500hpa/wind_500hpa', 'u', 500), # 500
             'v_component_of_wind': ('/mnt/data/sonia/cyclone/0.25/wind_500hpa/wind_500hpa', 'v', 500), # 500
             'temperature': ('/mnt/data/sonia/cyclone/0.25/temperature/temperature', 't', 925), # 925
             'specific_humidity': ('/mnt/data/sonia/cyclone/0.25/humidity/humidity', 'q', 500), # 500
            }

out_dir = '/mnt/data/sonia/climax-data/train-raw'
os.makedirs(out_dir, exist_ok=True)


lats = np.linspace(90, -90, 128)
lons = np.linspace(0, 360, 256, endpoint=False)

# for var_name, (var_path, short_name, level) in variables.items():
#     os.makedirs(os.path.join(out_dir, var_name), exist_ok=True)
#     for yr in tqdm(range(2016, 2025)):
#         ds = xr.open_dataset(f'{var_path}.{yr}.nc')
#         ds = ds[[short_name]].interp(lat=lats, lon=lons, method="linear")
#         if level is not None:
#             ds = ds.sel(pressure_level=level)
#             if 'pressure_level' not in ds.dims:
#                 ds = ds.expand_dims('pressure_level')
#             ds = ds.transpose("time", "pressure_level", "lat", "lon")
#             ds = ds.rename({'pressure_level': 'level'})
#         else:
#             ds = ds.transpose("time", "lat", "lon")
#         correct_time = ds['time'].values[0] + pd.to_timedelta(np.arange(ds.dims['time']) * 6, unit='h')
#         ds = ds.assign_coords(time=correct_time) # incase it wasn't read in as 6hrly
#         print(var_name, ds.to_array().shape, ds)
#         ds.to_netcdf(os.path.join(out_dir, var_name, f'{var_name}_{yr}.nc'))


# constants file
c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            'land_sea_mask',
            'geopotential', # ERA5 uses surface geopotential to represent orography
        ],
        'year': '1940',
        'month': '01',
        'day': '01',
        'time': '00:00',
        'format': 'netcdf',
        'grid': '1.40625/1.40625', 
    },
    os.path.join(out_dir, 'constants_raw.nc')
)
ds = xr.open_dataset(os.path.join(out_dir, 'constants_raw.nc')).load()

# Drop time bounds if they exist
if "time" in ds.dims:
    ds = ds.squeeze("time").drop_vars("time", errors="ignore")
if "valid_time" in ds.coords:
    ds = ds.drop_vars("valid_time", errors="ignore")

ds = ds.rename({
    "latitude": "lat",
    "longitude": "lon",
    "z": "orography"
})

# remove polar coords
ds = ds.interp(lat=lats, lon=lons, method="nearest")

lat_1d = ds["lat"].values
lon_1d = ds["lon"].values
lat2d_grid = np.repeat(lat_1d[:, np.newaxis], len(lon_1d), axis=1)
ds["lat2d"] = (("lat", "lon"), lat2d_grid)

ds.to_netcdf(os.path.join(out_dir, 'constants.nc'))
print(ds)
print(f"Successfully formatted constants.nc with shape: {ds.dims}")

os.remove(os.path.join(out_dir, 'constants_raw.nc'))
