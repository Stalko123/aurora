import xarray as xr
import datetime
from aurora import Batch, Metadata
import torch
# A dictionary that maps AURORA variable names to ERA5 variable names for atmospheric/surfacic data
aurora_to_era5_atm = {
    't': 'temperature',
    }

aurora_to_era5_surf = {
    '10u': '10m_u_component_of_wind',
    '10v': '10m_v_component_of_wind',
    'swh': 'significant_height_of_combined_wind_waves_and_swell',
    'mwd': 'mean_wave_direction',
    'mwp': 'mean_wave_period',
    
    
}

aurora_to_era5 = {'2t': '2m_temperature',
    '10u': '10m_u_component_of_wind',
    '10v': '10m_v_component_of_wind',
    'swh': 'significant_height_of_combined_wind_waves_and_swell',
    'mwd': 'mean_wave_direction',
    'mwp': 'mean_wave_period',
    'pp1d': 'peak_wave_period',
    'shww': 'significant_height_of_wind_waves',
    'mdww': 'mean_direction_of_wind_waves',
    'mpww': 'mean_period_of_wind_waves',
    'shts': 'significant_height_of_total_swell',
    'mdts': 'mean_direction_of_total_swell',
    'mpts': 'mean_period_of_total_swell',
    'swh1': 'significant_wave_height_of_first_swell_partition',
    'mwd1': 'mean_wave_direction_of_first_swell_partition',
    'mwp1': 'mean_wave_period_of_first_swell_partition',
    'swh2': 'significant_wave_height_of_second_swell_partition',
    'mwd2': 'mean_wave_direction_of_second_swell_partition',
    'mwp2': 'mean_wave_period_of_second_swell_partition',
    '10u_wave': 'u_component_stokes_drift',
    '10v_wave': 'v_component_stokes_drift',
    'wind': 'ocean_surface_stress_equivalent_10m_neutral_wind_speed','t': 'temperature',
    'u': 'u_component_of_wind',
    'v': 'v_component_of_wind',
    'q': 'specific_humidity',
    'z': 'geopotential_at_surface',
    'slt': 'soil_type',
    'lsm' :'land_sea_mask'}


niveaux=[50]


def get_two_timesteps_era5(path, var_name, year, month, day, hour): #Works good gives a dataset with two convecutive timesteps corresponding to the requested variable.
    
        ds = xr.open_zarr(path,
                        chunks=None, 
                        storage_options=dict(token='anon'),
                        decode_times=True)
            
        analysis_ready = ds.sel(time=slice(ds.attrs['valid_time_start'], ds.attrs['valid_time_stop']))
            
        query_time_start = datetime.datetime(year, month, day, hour)
        query_time_end = query_time_start + datetime.timedelta(hours=6)
            
        analysis_ready = ds.sel(time=[query_time_start, query_time_end], method='nearest')

        if var_name in ['t','u','v','q']:
            T=analysis_ready[aurora_to_era5[var_name]]
            T = T[:, :len(niveaux) , : , :]
            return T
        if var_name=='z':
            T=analysis_ready[aurora_to_era5[var_name]]
            T = T.expand_dims({'level': len(niveaux)}, axis=2)
            dims_order = [T.dims[0], T.dims[2], T.dims[1], T.dims[3]]
            T = T.transpose(*dims_order)
            return T

        return analysis_ready[aurora_to_era5[var_name]]


def get_metadata_era5(path, year, month, day, hour_start): #Works good gives a dataset with two convecutive timesteps corresponding to the requested variable.

    # Open the dataset to get metadata
    ds = xr.open_zarr(path,
                        chunks={'time': 48},
                        storage_options=dict(token='anon'),
                        decode_times=True)

    analysis_ready = ds.sel(time=slice(ds.attrs['valid_time_start'], ds.attrs['valid_time_stop']))
            
    query_time_start = datetime.datetime(year, month, day, hour_start)
    query_time_end = query_time_start + datetime.timedelta(hours=6)
            
    analysis_ready = ds.sel(time=[query_time_start, query_time_end], method='nearest')
    # Extract metadata
    valid_time_start = analysis_ready.attrs['valid_time_start']
    valid_time_stop = analysis_ready.attrs['valid_time_stop']
    '''levels = tuple(analysis_ready['level'].values) if 'level' in analysis_ready else None'''
    levels= tuple(niveaux)
    latitudes = torch.from_numpy(analysis_ready['latitude'].values) if 'latitude' in analysis_ready else None
    longitudes = torch.from_numpy(analysis_ready['longitude'].values) if 'longitude' in analysis_ready else None
    # Create a Metadata object
    query_time_start = datetime.datetime(year, month, day, hour_start)
    query_time_end = query_time_start + datetime.timedelta(hours=6)

    metadata = Metadata(

        time= tuple([query_time_end]),
        atmos_levels=levels,
        lat=latitudes,
        lon=longitudes,
        rollout_step=0
    )

    return metadata

def get_static_vars_era5(path): #This function is not used in the current implementation but can be useful for future extensions.
    # Open the dataset to get static variables
    ds = xr.open_zarr(path,
                        chunks=None, 
                        storage_options=dict(token='anon'),
                        decode_times=True)
            
    query_time_start = datetime.datetime(2025, 1, 1, 0)


    ds = ds.sel(time=[query_time_start],method='nearest')  # Select a time to access static variables

    # Extract static variables

    return (torch.from_numpy(ds[aurora_to_era5['lsm']].values)[0], torch.from_numpy(ds[aurora_to_era5['slt']].values)[0],torch.from_numpy(ds[aurora_to_era5['z']].values)[0],torch.from_numpy(ds['land_sea_mask'].values)[0],torch.from_numpy(ds['model_bathymetry'].values)[0])  # Assuming 'land_sea_mask' is the static variable of interest

    
def get_batch_era5(path, year, month, day, hour_start): #Works good gives a dataset with two convecutive timesteps corresponding to the requested variable.
    
    surface_vars= { key:0 for key in list(aurora_to_era5_surf.keys()) }
    atm_vars = { key:0 for key in list(aurora_to_era5_atm.keys()) }
    
   
    for key in surface_vars.keys():
        # Open the dataset for the specific variable
       
        surf_vars_ds = get_two_timesteps_era5(path, key, year, month, day, hour_start)
        surface_vars[key] = torch.from_numpy(surf_vars_ds.values[:2][None])  # Convert to torch tensor and add a new dimension
       
    
    
    for key in atm_vars.keys():
        # Open the dataset for the specific variable

     
        atm_vars_ds = get_two_timesteps_era5(path, key, year, month, day, hour_start)
        
        atm_vars[key] = torch.from_numpy(atm_vars_ds.values[:2][None])
        

    lsm,slt,z,lat_mask,wmb = get_static_vars_era5(path)  # Get static variables if needed
    
    batch = Batch(
        surf_vars=surface_vars,
        atmos_vars=atm_vars,
        metadata=get_metadata_era5(path, year, month, day, hour_start),  # Get metadata for the batch
        static_vars= {
           
            'lsm': lsm   # Land-sea mask
            ,'slt': slt   # Soil type
            ,'z': z       # Geopotential at surface
            ,'lat_mask': lat_mask  # Latitude mask
            ,'wmb': wmb   # Wave mask or other static variable if needed
        }
    )

    return batch
        



if __name__ == "__main__":
    import pickle
    from aurora import AuroraWave

    # Example usage
    path = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
    year = 2020
    month = 1
    day = 1
    hour = 0
    var_name = '2t'  # Example variable name

    with open("saved_batch.pkl", "wb") as f:
        batch= get_batch_era5(path, year, month, day, hour)

    model = AuroraWave(surf_vars=tuple(list(aurora_to_era5_surf.keys())),  # Surface variables
        angle_surf_vars= ['mwd', 'mdww','mdts', 'mwd1', 'mwd2'],  # Angle surface variables
    )
    print("Model created successfully.")
    
    model.eval()
    print("Feeding the batch to the model...")
    '''import pdb
    pdb.set_trace()'''
    pred = model.forward(batch)

    print("Prediction completed.")
