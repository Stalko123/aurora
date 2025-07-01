#I want to create a function that takes two paths for two concecutive timesteps in the era5 dataset and return a batch with all the variables used by the AuroraWave model, knowing that the path point to a .zarr file. The file takes in argument even the geographical zone in format of the index of the latitude and longiutude or the values of longitude and latitude themselves.

import xarray as xr
import numpy as np
import datetime


# Variable groupings for AuroraWave
SURF_VARS = [
    "2t", "10u", "10v", "swh", "mwd", "mwp", "pp1d", "shww", "mdww", "mpww", "shts", "mdts", "mpts",
    "swh1", "mwd1", "mwp1", "swh2", "mwd2", "mwp2", "10u_wave", "10v_wave", "wind"
]
STATIC_VARS = ["lsm", "slt", "z", "wmb", "lat_mask"]
ATMOS_VARS = ["t", "u", "v", "q", "z"]

def get_aurorawave_batch_from_era5(
    path,
    lat_idx=None,
    lon_idx=None,
    lat=None,
    lon=None,
    variables=None,
):
    """
    Loads a batch for AuroraWave model from the first two timesteps of a daily ERA5 .zarr file.
    Returns a dict for t0 and t1, each with surf_vars, static_vars, and atmos_vars.
    """
    if variables is None:
        variables = SURF_VARS + STATIC_VARS + ATMOS_VARS

    ds = xr.open_zarr(path)

    if lat_idx is None or lon_idx is None:
        if lat is not None and lon is not None:
            lat_vals = ds['latitude'].values
            lon_vals = ds['longitude'].values
            lat_idx = np.argmin(np.abs(lat_vals - lat))
            lon_idx = np.argmin(np.abs(lon_vals - lon))
        else:
            raise ValueError("Either lat_idx/lon_idx or lat/lon must be provided.")

    def extract_vars(time_idx):
        surf = {}
        stat = {}
        atmos = {}
        for var in variables:
            if var not in ds:
                continue
            if "time" in ds[var].dims:
                arr = ds[var].isel(time=time_idx, latitude=lat_idx, longitude=lon_idx).values
            else:
                arr = ds[var].isel(latitude=lat_idx, longitude=lon_idx).values
            if var in SURF_VARS:
                surf[var] = arr
            elif var in STATIC_VARS:
                stat[var] = arr
            elif var in ATMOS_VARS:
                atmos[var] = arr
        return {"surf_vars": surf, "static_vars": stat, "atmos_vars": atmos}

    return {"t0": extract_vars(0), "t1": extract_vars(1)}

def get_aurorawave_batch_from_era5_cloud(
    year,
    month,
    day,
    lat_idx=None,
    lon_idx=None,
    lat=None,
    lon=None,
    variables=None,
    google_path='gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
):
    """
    Fetches ERA5 data from Google Cloud for a given day and returns a batch in Aurora format for the first two timesteps.
    """
    if variables is None:
        variables = SURF_VARS + STATIC_VARS + ATMOS_VARS

    ds = xr.open_zarr(google_path, chunks=None, storage_options=dict(token='anon'), decode_times=True)

    # Select the day
    date0 = datetime.datetime(year, month, day, 0)
    date1 = datetime.datetime(year, month, day, 1)
    ds_day = ds.sel(time=[date0, date1], method='nearest')

    if lat_idx is None or lon_idx is None:
        if lat is not None and lon is not None:
            lat_vals = ds['latitude'].values
            lon_vals = ds['longitude'].values
            lat_idx = np.argmin(np.abs(lat_vals - lat))
            lon_idx = np.argmin(np.abs(lon_vals - lon))
        else:
            raise ValueError("Either lat_idx/lon_idx or lat/lon must be provided.")

    def extract_vars(time_idx):
        surf = {}
        stat = {}
        atmos = {}
        for var in variables:
            if var not in ds_day:
                continue
            if "time" in ds_day[var].dims:
                arr = ds_day[var].isel(time=time_idx, latitude=lat_idx, longitude=lon_idx).values
            else:
                arr = ds_day[var].isel(latitude=lat_idx, longitude=lon_idx).values
            if var in SURF_VARS:
                surf[var] = arr
            elif var in STATIC_VARS:
                stat[var] = arr
            elif var in ATMOS_VARS:
                atmos[var] = arr
        return {"surf_vars": surf, "static_vars": stat, "atmos_vars": atmos}

    return {"t0": extract_vars(0), "t1": extract_vars(1)}

