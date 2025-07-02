import xarray as xr
import datetime

# A dictionary that maps AURORA variable names to ERA5 variable names for atmospheric/surfacic data
aurora_to_era5_atm = {
    't': 'temperature',
    'u': 'u_component_of_wind',
    'v': 'v_component_of_wind',
    'q': 'specific_humidity',
    'z': 'geopotential_at_surface'}

aurora_to_era5_surf = {
    '2t': '2m_temperature',
    '10u': '10m_u_component_of_wind',
    '10v': '10m_v_component_of_wind',
    'swh': 'significant_height_of_combined_wind_waves_and_swell',
    'mwd': 'mean_wave_direction',
    'mwp': 'mean_wave_period',
    'pp1d': 'peak_wave_period',
    'shww': 'significant_height_of_wind_waves',
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
    'wind': 'ocean_surface_stress_equivalent_10m_neutral_wind_speed',
}


def get_two_timesteps_era5(path, var_name, year, month, day, hour): #Works good gives a dataset with two convecutive timesteps corresponding to the requested variable.
    
    ds = xr.open_zarr(path,
                      chunks=None, 
                      storage_options=dict(token='anon'),
                      decode_times=True)
        
    analysis_ready = ds.sel(time=slice(ds.attrs['valid_time_start'], ds.attrs['valid_time_stop']))
        
    query_time_start = datetime.datetime(year, month, day, hour)
    query_time_end = query_time_start + datetime.timedelta(hours=6)
        
    analysis_ready = ds.sel(time=[query_time_start, query_time_end], method='nearest')

    return analysis_ready[aurora_to_era5_surf[var_name]]





if __name__ == "__main__":
    # Example usage
    google_path = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
    year = 2020
    month = 1
    day = 1
    hour = 0
    var_name = '2t'  # Example variable name

    result = get_two_timesteps_era5(google_path, var_name, year, month, day, hour)
    print(result)  # This will print the result of the function call